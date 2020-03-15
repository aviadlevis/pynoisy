# Program to calculate the Q-metric and error for an example simulation of closure phases from a GRMHD movie and static source. The data set is divided in multiple segments and Q is calculated using equations 21 and 26-28 of Roelofs et al. (2017). The outcome with the example data set is Figure 6 of Roelofs et al. (2017).
# Maciek Wielgus, 2017/11/28, maciek.wielgus@gmail.com
# Dom Pesce, 2018/08/02, dom.pesce@gmail.com
# based on Freek Roelofs et al., ApJ 847, Quantifying Intrinsic Variability of Sagittarius A* Using Closure Phase Measurements of the Event Horizon Telescope
import numpy as np
import matplotlib.pyplot as plt

def qmetric(time, obs, obs_err, bintime=0, segtime=0, diftime=0, product='cphase',detrend_deg=-1,diff_accuracy = 0.1,sigma_cut=500,loess=False):
    """Main function to calculate q-metric and error
    bintime - binning period for unevenly sampled data
    defaultly calculated as median difference between observations
    segtime - duration of segment for detrending purposes
    diftime - distance between points used for differentiation
    product - what statistics
    detrend_deg - degree of polynomial used for segment-wise detrending
    -1 = no detrending, 0 = constant, 1 = linear, ...
    diff_accuracy = 
    """

    time=time[obs_err<sigma_cut]
    obs=obs[obs_err<sigma_cut]
    obs_err=obs_err[obs_err<sigma_cut]
    #CHECK IF DATA IS SAMPLED UNIFORMLY
    #IF IT IS, binning==False
    median_diff = np.median(np.diff(time))
    if bintime==0:
            bintime = median_diff#estimate bintime
    binning = not all((np.diff(time) - np.mean(np.diff(time)))==0)
    #IF UNUNIFORMLY SAMPLED DATA DO THE BINNING
    if binning==True:
        #print('Non-uniform sampling detected, binning the data with bintime = %s' %str(bintime))
        bins = np.arange(min(time)-bintime/2., max(time)+bintime+1., bintime)
        digitized = np.digitize(time, bins, right=False) # Assigns a bin number to each of the closure phases
        bin_times = []
        bin_means = []
        bin_errors = []
        for bin in range(len(bins)+1):
            if len(obs[digitized==bin])>0:
                mean_local = circular_mean_weights(obs[digitized==bin],obs_err[digitized==bin])
                err_local = np.sqrt(np.sum(obs_err[digitized==bin]**2))/len(obs_err[digitized==bin])
                bin_times.append(0.5*(bins[bin-1] + bins[bin]) )
                bin_means.append(mean_local)
                bin_errors.append(err_local)
        time = np.asarray(bin_times)
        obs = np.asarray(bin_means)
        obs_err= np.asarray(bin_errors)
    else:
        print('Data uniformly sampled, no binning')

    #DIFFERENTIATING THE DATA
    if diftime>0:
        time,obs, obs_err = diff_time(time,obs,obs_err, diftime, accuracy = diff_accuracy)
        #print(obs)

    #SEGMENTATION OF DATA
    if segtime==0:
        if detrend_deg>-1:
            if loess==True:
                obs = loess(time,obs,obs_err,time,width=10.0,order=detrend_deg)
            else:
                obs = detrending_polyfit(time,obs,obs_err,detrend_deg)
        q, dq, n = find_q_basic(obs,obs_err)
    else:
        print('Segmenting data with segtime = %s' %str(segtime))
        segments = np.arange(min(time), max(time)+segtime, segtime)
        digitized = np.digitize(time, segments, right=False) # Assigns a segment number to each measurement
        time_segments = []
        obs_segments = []
        obs_err_segments = []
        for cou in range(1,len(segments)+1):
            time_local = time[digitized==cou]
            obs_local = obs[digitized==cou]
            obs_err_local = obs_err[digitized==cou]
            N = 10.
            if len(time_local)>N: #let's have at least N datapoints in each segment
                time_segments.append(time_local)
                if detrend_deg>-1:
                    obs_local = detrending_polyfit(time_local,obs_local,obs_err_local,detrend_deg)
                obs_segments.append(obs_local)
                obs_err_segments.append(obs_err_local)
        #Calculate q metric in each segment
        sig2_segments = []
        eps2_segments = []
        n_segments = []
        tot_segments = len(time_segments)
        #print(time_segments)
        for cou in range(tot_segments):
            #q_loc, dq_loc, n_loc = find_q_basic(obs_segments[cou],obs_err_segments[cou])
            sig_loc, eps_loc, n_loc = find_sig_eps_basic(obs_segments[cou],obs_err_segments[cou],product)
            sig2_segments.append(sig_loc**2)
            eps2_segments.append(eps_loc**2)
            n_segments.append(float(n_loc))
        sig2_global = np.average(np.asarray(sig2_segments),weights=np.asarray(n_segments))
        eps2_global = np.average(np.asarray(eps2_segments),weights=np.asarray(n_segments))
        n_tot = np.sum(np.asarray(n_segments))
        q = (sig2_global - eps2_global)/sig2_global
        dq =  np.sqrt(2./n_tot)*eps2_global*np.sqrt(np.average( (np.asarray(sig2_segments))**2,weights=np.asarray(n_segments)))/sig2_global**2 
    return q, dq

def find_q_basic(obs,obs_err,product='cphase'):
    #most basic function to get q metric
    #no binning or detrending
    obs = np.asarray(obs)
    obs_err = np.asarray(obs_err)
    if product=='cphase':
        obs_sigma = circular_std(obs)
        eps_thermal = eps_analytic(obs_err)
        #eps_thermal = eps_MC(obs_err)
    elif product=='camplitude':
        obs_sigma = 1.#place holder
        eps_thermal = 1.#place holder
    n = len(obs)
    if n > 0:
        q=(obs_sigma**2- eps_thermal**2)/obs_sigma**2
        dq = np.sqrt(2./n)*(1-q)
    else:
        q = np.nan
        dq = np.nan
    return q, dq, n
    

def find_sig_eps_basic(obs,obs_err,product='cphase'):
    #most basic function to get q metric
    #no binning or detrending
    obs = np.asarray(obs)
    obs_err = np.asarray(obs_err)
    if product=='cphase':
        obs_sigma = circular_std(obs)
        eps_thermal = eps_analytic(obs_err)
    elif product=='camplitude':
        obs_sigma = np.std(obs)#place holder
        eps_thermal = (np.sum(obs_err))/len(obs_err) #place holder
    n = len(obs)
    return obs_sigma, eps_thermal, n

def eps_MC(err):
    deg2rad = np.pi/180.
    rad2deg = 180./np.pi
    """Calculate tilde{epsilon} using Monte Carlo approach assuming Gaussian errors"""
    N = int(1e3)
    epsilons=np.zeros(N)
    for i in range(N):
        this_it=np.zeros(len(err))
        for j in range(len(err)):
            this_it[j]=np.random.normal(0.,err[j]*deg2rad)
        #print(this_it)
        this_cosi=[np.cos(x) for x in this_it]
        this_sini=[np.sin(x) for x in this_it]
        this_cos_avg=np.mean(this_cosi)
        this_sin_avg=np.mean(this_sini)
        this_R=np.sqrt(this_sin_avg**2+this_cos_avg**2)
        #print('R= ',this_R)
        this_obs_sigma=np.sqrt(-2.*np.log(this_R))*rad2deg
        #print('eps= ',this_obs_sigma)
        epsilons[i]=this_obs_sigma
    epsi=np.mean(epsilons)
    return epsi

def eps_analytic(err):
    #err in deg
    err = np.asarray(err)
    R = np.mean( np.exp(-(err*np.pi/180.)**2/2.) )
    eps = np.sqrt(-2*np.log(R))*180./np.pi
    return eps

def stR(theta):
#i/o in degrees
    theta = np.asarray(theta)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    R = np.sqrt(C**2+S**2)
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi
    return st,R

def R_from_st(st):
    #st in deg
    R = np.exp(-(st*np.pi/180.)**2/2.)
    return R

def detrending_polyfit(time,obs,errs=None,deg=1,weights=True):
    #just subtract linear fit inside segment
    time = np.asarray(time)
    t0 = time[0]
    time = time-t0
    obs = np.asarray(obs)
    #print 'OBS BEFORE'
    #print obs
    #plt.scatter(time, obs)
    obs = obs*np.pi/180

    obs = np.unwrap(obs)
    #plt.scatter(time, obs)
    we = 0*obs+1.#just ones
    if weights==True:
        we= 1/errs
    
    fit = np.polyfit(time, obs, deg,w=we)
    obs = obs - np.polyval(fit,time)
    #plt.scatter(time, np.polyval(fit,time))
    #plt.scatter(time, obs)
    obs = (( obs + np.pi) % (2 * np.pi ) - np.pi)*180./np.pi
    #print 'OBS AFTER'
    #print obs
    #plt.scatter(time, obs)
    #plt.show()
    return obs

def diff_time(time,obs,err, dt, accuracy = 0.1):
    time_new = []
    obs_new = []
    err_new = []
    time = np.asarray(time)
    for cou in range(len(time)):
        delta = np.abs(time[cou]+dt - time)
        ind = np.argmin(delta)
        if delta[ind]<accuracy*dt:
            time_new.append(time[cou])
            obs_new.append(180/np.pi*np.angle(np.exp(1j*np.pi/180.*(obs[ind]-obs[cou]))))
            err_new.append( np.sqrt(err[ind]**2+err[cou]**2) )
    return np.asarray(time_new),np.asarray(obs_new),np.asarray(err_new)


def inflate_noise(datfile, inflation_factor, savefile = ''):
    #inflates noise by factor sqrt(inflation_factor**2 + 1) 
    data=np.loadtxt(datfile)    
    time=data[:,0] #units? #HOURS
    obs=data[:,1] #time series (e.g. closure phase)
    obs_err=data[:,2] #obs measurement error

    obs_err = np.asarray(obs_err)*np.sqrt(1.+inflation_factor**2)
    obs_noise = [np.random.normal(0.,obs_err[x]) for x in range(len(obs_err))]
    obs = np.asarray(obs) + np.asarray(obs_noise)
    data = np.asarray([time, obs, obs_err])
    if savefile=='':
        savefile=datfile
    np.savetxt(savefile,np.transpose(data))

def rescale_noise(datfile, rescale_sd, savefile = ''):
    #adds noise, possibly with sigma given in the file 
    data=np.loadtxt(datfile)    
    time=np.asarray(data[:,0]) #units? #HOURS
    obs=np.asarray(data[:,1]) #time series (e.g. closure phase)
    obs_err=data[:,2] #obs measurement error
    obs_err=np.asarray(obs_err)*rescale_sd
    data = np.asarray([time, obs, obs_err])
    if savefile=='':
        savefile=datfile
    np.savetxt(savefile,np.transpose(data))

def add_noise(datfile, noise_sd, savefile = '',copy_sigmas=False):
    #adds noise, possibly with sigma given in the file 
    data=np.loadtxt(datfile)    
    time=data[:,0] #units? #HOURS
    obs=data[:,1] #time series (e.g. closure phase)
    obs_err=data[:,2] #obs measurement error
    if copy_sigmas==False:
        obs_err = [noise_sd]*len(obs_err)
        obs_noise = np.random.normal(0.,noise_sd,len(obs))
    else:
        obs_err=np.asarray(obs_err)
        obs_noise = [np.random.normal(0.,noise_sd*obs_err[x]) for x in range(len(obs_err))]
    obs = np.asarray(obs) + np.asarray(obs_noise)
    data = np.asarray([time, obs, obs_err])
    if savefile=='':
        savefile=datfile
    np.savetxt(savefile,np.transpose(data))

    
def add_noise_file(datfile, noise_sd=1., errfile='', savefile = '',copy_sigmas=False):
    #inflates noise by factor sqrt(inflation_factor**2 + 1) 
    data=np.loadtxt(datfile)    
    time=data[:,0] #units? #HOURS
    obs=data[:,1] #time series (e.g. closure phase)
    if errfile=='':
        errfile = datfile
    err_foo=np.loadtxt(errfile)
    obs_err=err_foo[:,2] #obs measurement error
    if copy_sigmas==False:
        obs_err = [noise_sd]*len(obs_err)
        obs_noise = np.random.normal(0.,noise_sd,len(obs))
    else:
        obs_err=np.asarray(obs_err)
        obs_noise = [np.random.normal(0.,noise_sd*obs_err[x]) for x in range(len(obs_err))]
    obs = np.asarray(obs) + np.asarray(obs_noise)
    data = np.asarray([time, obs, obs_err])
    if savefile=='':
        savfile=datfile
    np.savetxt(savefile,np.transpose(data))


#####---------------------------------
#FUNCTIONS FOR CALCULATING STATISTICS
#####---------------------------------

def circular_mean_weights(theta,err='ones'):
    #i/o in degrees
    theta = np.asarray(theta)*np.pi/180. #to radians
    if str(err)=='ones':
        err = 0.*theta+1.
    err = np.asarray(err)*np.pi/180. #to radians
    weights = 1./np.asarray(err**2)
    C = np.average(np.cos(theta),weights=weights)
    S = np.average(np.sin(theta),weights=weights)
    mean = np.arctan2(S,C)*180./np.pi
    return mean

def circular_mean(theta):
    #i/o in degrees
    theta = np.asarray(theta)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    mt = np.arctan2(S,C)*180./np.pi
    return mt

def circular_std(theta):
    #i/o in degrees
    theta = np.asarray(theta)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi
    return st

def circular_std_of_mean(theta):
    #i/o in degrees
    theta = np.asarray(theta)*np.pi/180.
    C = np.mean(np.cos(theta))
    S = np.mean(np.sin(theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi/np.sqrt(len(theta))
    return st

def diff_side(x):
    x = np.asarray(x)
    xp = x[1:]
    xm = x[:-1]
    xdif = xp-xm
    dx = np.angle(np.exp(1j*xdif*np.pi/180.))*180./np.pi
    return dx

def circular_std_dif(theta):
    theta = np.asarray(theta)*np.pi/180.
    dif_theta = diff_side(theta)
    C = np.mean(np.cos(dif_theta))
    S = np.mean(np.sin(dif_theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi/np.sqrt(2.)
    return st

def circular_std_of_mean_dif(theta):
    theta = np.asarray(theta)*np.pi/180.
    dif_theta = diff_side(theta)
    C = np.mean(np.cos(dif_theta))
    S = np.mean(np.sin(dif_theta))
    st = np.sqrt(-2.*np.log(np.sqrt(C**2+S**2)))*180./np.pi/np.sqrt(len(theta))/np.sqrt(2.)
    return st


def diagnostic_plots(filepath, detrend_deg=-1,diff_show=0.01,errscale=1.,sigma_cut=500,segtime=0):
    import matplotlib.pyplot as plt
    data = np.loadtxt(filepath)
    time= data[:,0]
    cphase=data[:,1]
    sigmacp = data[:,2]
    time=time[sigmacp<sigma_cut]
    cphase=cphase[sigmacp<sigma_cut]
    sigmacp=sigmacp[sigmacp<sigma_cut]
    fig,ax = plt.subplots(2,2)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    ####UP LEFT PANEL
    ax[0,0].errorbar(time,cphase,errscale*sigmacp,color='blue',label='raw',fmt='o')
    if detrend_deg>-1:
        cpdtr = detrending_polyfit(time,cphase,sigmacp,detrend_deg)
        ax[0,0].errorbar(time,cpdtr,errscale*sigmacp,color='r',label='detrended, deg= '+str(detrend_deg),fmt='o')
    ax[0,0].grid()
    ax[0,0].set_xlabel('time')
    
    ax[0,0].legend()
    #plt.show()

    ####UP RIGHT PANEL
    ax[0,1].errorbar(time,cphase,errscale*sigmacp,color='blue',label='raw',fmt='o')
    time2,obs2, obs_err2 = diff_time(time,cphase,sigmacp, diff_show)

    ax[0,1].errorbar(time2,obs2,errscale*obs_err2,color='r',label='differentiated, dt= '+str(diff_show),fmt='o')
    ax[0,1].set_xlabel('time')
    ax[0,1].grid()
    ax[0,1].legend()
    

    ####BOTTOM LEFT PANEL
    variab = [-1,0,1,2,3,4,5,6,7]
    err= np.zeros(len(variab))
    qm= np.zeros(len(variab))
    for cou in range(len(variab)):
        #segtime=segsp[cou]
        foo = qmetric(time,cphase,sigmacp,detrend_deg=variab[cou],segtime=segtime)
        qm[cou]=foo[0]
        err[cou]=foo[1]
    #qm[cou],foo = qf.qmetric(filep,diftime=variab[cou])
    ax[1,0].errorbar(variab,qm,err)
    ax[1,0].set_xlabel('poly degree')
    ax[1,0].set_ylabel('Q')
    #plt.show()


   ####BOTTOM RIGHT PANEL
    variab = np.linspace(0.5*diff_show,2.*diff_show,12)
    err= np.zeros(len(variab))
    qm= np.zeros(len(variab))
    for cou in range(len(variab)):
        #segtime=segsp[cou]
        foo = qmetric(time,cphase,sigmacp,diftime=variab[cou],segtime=segtime)
        qm[cou]=foo[0]
        err[cou]=foo[1]
    #qm[cou],foo = qf.qmetric(filep,diftime=variab[cou])
    ax[1,1].errorbar(variab,qm,err)
    ax[1,1].set_xlabel('difftime')
    #ax[1,1].ticks_params(axis='x',rotation='vertical')
    #ax[1,1].set_ylabel('Q')
    plt.show()

def loess(t,X,X_err,t_new,width=10.0,order=1):
        """Determine the loess regression for a time series.
        
        Args:
	   t (Array): the input time array
	   X (Array): the input data array
	   X_err (Array): the input data uncertainties
	   t_new (Array): the time array over which to perform the loess regression; assumed to have the same units as t
	   width (float): the window over which to perform the regression; assumed to have the same units as t
	   order (int): the order of the polynomial to be fit across the window at each t_new value
        
        Returns:
	   X_new: a data array sampled at 
        
	"""
        
        X_new = np.zeros_like(t_new)
        
        for i in range(len(t_new)):
                
                center = t_new[i]
                
                if ((t_new[i] - (width/2.0)) < t[0]):
                        start = t[0]
                        finish = center + (width / 2.0)
                        
                if (((t_new[i] - (width/2.0)) >= t[0]) & ((t_new[i] + (width/2.0)) <= t[-1])):
                        start = center - (width / 2.0)
                        finish = start + width
                        
                if ((t_new[i] + (width/2.0)) > t[-1]):
                        start = center - (width / 2.0)
                        finish = t[-1]
            
                mask = ((t >= start) & (t <= finish))
                
                t_here = t[mask]
                X_here = X[mask]
                
                weight = (1.0 - ((np.abs(t_here - center)/(width/2.0))**3.0))**3.0
                weight *= 1.0/X_err[mask]
		
                coeffs = np.polyfit(t_here,X_here,deg=order,w=weight)
                
                for j in range(len(coeffs)):
                        X_new[i] += coeffs[len(coeffs)-j-1]*(center**float(j))
                        
        return X_new
