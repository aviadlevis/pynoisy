import ehtim as eh
import ehtim.const_def as ehc
import numpy as np
import os
from scipy.ndimage import median_filter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pynoisy


def xarray_to_hdf5(movie, obs, fov, flipy=True):
    """Transform xarray to and HDF5 movie

    Args:
        movie (xr.DataArray): a dataarray with dims=['t', 'x', 'y']
        obs (Obsdata):  observation object
        fov (float): Field of view in micro arc secs (UAS)
        flipy (bool): Flip y-axis due to ehtim y axis flip

    Returns:
        output (Movie): a ehtim movie object
    """
    mjd = obs.mjd  # modified julian date of observation
    ra  = obs.ra   # ra of the source
    dec = obs.dec  # dec of the source
    rf  = obs.rf   # reference frequency observing at corresponding to 1.3 mm wavelength

    start_time = obs.tstart
    end_time = obs.tstop
    num_frames = movie.sizes['t']
    times = np.linspace(start_time, end_time, num_frames)

    im_list = []
    for i, time in enumerate(times):
        frame = movie.isel(t=i)
        image = eh.image.make_empty(frame.sizes['x'], fov * ehc.RADPERUAS, ra, dec, rf, mjd=mjd, source=obs.source)
        image.time = time
        image.ivec = np.flipud(frame).ravel() if flipy else frame.data.ravel()
        im_list.append(image)

    return eh.movie.merge_im_list(im_list)

def load_obs(array_path, uvfits_path):
    """Load observations.

    Args:
        array_path (str): path to array txt file.
        uvfits_path (str): path to the observation file.

    Returns:
        obs (Obsdata): observation object
    """
    obs = eh.obsdata.load_uvfits(uvfits_path, remove_nan=True)

    # Load telescope site locations and SEFDs (how noisy they are)
    eht = eh.array.load_txt(array_path)

    # Copy the correct mount types
    t_obs = list(obs.tarr['site'])
    t_eht = list(eht.tarr['site'])
    t_conv = {'AA': 'ALMA', 'AP': 'APEX', 'SM': 'SMA', 'JC': 'JCMT', 'AZ': 'SMT', 'LM': 'LMT', 'PV': 'PV', 'SP': 'SPT'}
    for t in t_conv.keys():
        if t in obs.tarr['site']:
            for key in ['fr_par', 'fr_elev', 'fr_off']:
                obs.tarr[key][t_obs.index(t)] = eht.tarr[key][t_eht.index(t_conv[t])]

    return obs

def ehtim_movie(frames, obs_sgra, total_flux=2.23, normalize_flux=True, fov=125, fov_units='uas',
                start_time=None, end_time=None, std_threshold=8, median_size=5,
                linpol_mag=0.3, linpol_corr=10.0, circpol_mag=0.1, cirpol_corr=5.0, seed=0):
    """Generate ehtim Movie object.

    Args:
        frames (list): A list of movie frames.
        obs_sgra (Obsdata): observation object.
        start_time (float): Start time of the movie. If None use SgrA observation start time.
        end_time (float): End time of the movie. If None use SgrA observation end time.
        total_flux (float): normalizing constant for the images.
        fov (float): field of view of the image in micro arcseconds.

    Returns:
        movie (Movie): An ehtim.Movie object containing normalized frames.
        obs (Obsdata): observation object.
    """
    mjd = obs_sgra.mjd  # modified julian date of observation
    ra = obs_sgra.ra  # ra of the source - sgra a*
    dec = obs_sgra.dec  # dec of the source - sgr a*
    rf = obs_sgra.rf  # reference frequency observing at corresponding to 1.3 mm wavelength

    if fov_units == 'uas':
        fov_scale = eh.RADPERUAS
    if fov_units == 'rad':
        fov_scale = 1.0
    fov *= fov_scale

    start_time = obs_sgra.tstart if start_time is None else start_time
    end_time = obs_sgra.tstop if end_time is None else end_time
    times = np.linspace(start_time, end_time, len(frames))

    movie_frames = []

    for frame in frames:
        median = median_filter(frame, size=median_size)
        mean = frame.mean() + 1e-10
        pixel_std = frame / mean
        frame[pixel_std > std_threshold] = median[pixel_std > std_threshold]

    flux_normalization = total_flux / (frames.mean(axis=0).sum() + 1e-10) if normalize_flux else 1.0
    for frame, time in zip(frames, times):
        im = eh.image.make_empty(frame.shape[0], fov, ra, dec, rf, source='SgrA')
        im.mjd = mjd
        im.time = time
        im.imvec = flux_normalization * frame.reshape(-1)
        im = im.add_random_pol(linpol_mag, linpol_corr*eh.RADPERUAS, circpol_mag, cirpol_corr*eh.RADPERUAS, seed=seed)
        movie_frames.append(im)

    movie = eh.movie.merge_im_list(movie_frames)

    # change the synthetic image coordinates to align with the obs
    movie.ra = obs_sgra.ra
    movie.dec = obs_sgra.dec
    movie.rf = obs_sgra.rf
    movie.reset_interp(interp='linear', bounds_error=False)
    return movie

def export_movie(im_List, out, fps=10, dpi=120, scale='linear', cbar_unit = 'Jy', gamma=0.5, dynamic_range=1000.0,
                 pad_factor=1, verbose=False):
    mjd_range = im_List[-1].mjd - im_List[0].mjd
    fig = plt.figure()

    extent = im_List[0].psize/eh.RADPERUAS*im_List[0].xdim*np.array((1,-1,-1,1)) / 2.
    maxi = np.max(np.concatenate([im.imvec for im in im_List]))

    unit = cbar_unit + '/pixel'

    if scale=='log':
        unit = 'log(' + cbar_unit + '/pixel)'

    if scale=='gamma':
        unit = '(' + cbar_unit + '/pixel)^gamma'

    def im_data(n):
        n_data = (n-n%pad_factor)//pad_factor
        if scale == 'linear':
            return im_List[n_data].imvec.reshape((im_List[n_data].ydim,im_List[n_data].xdim))
        elif scale == 'log':
            return np.log(im_List[n_data].imvec.reshape((im_List[n_data].ydim,im_List[n_data].xdim)) + maxi/dynamic_range)
        elif scale == 'gamma':
            return (im_List[n_data].imvec.reshape((im_List[n_data].ydim,im_List[n_data].xdim)) + maxi/dynamic_range)**(gamma)

    plt_im = plt.imshow(im_data(0), extent=extent, cmap=plt.get_cmap('afmhot'), interpolation='gaussian')
    if scale == 'linear':
        plt_im.set_clim([0,maxi])
    elif scale == 'log':
        plt_im.set_clim([np.log(maxi/dynamic_range),np.log(maxi)])
    elif scale == 'gamma':
        plt_im.set_clim([(maxi/dynamic_range)**gamma,(maxi)**(gamma)])

    plt.xlabel('Relative RA ($\mu$as)')
    plt.ylabel('Relative Dec ($\mu$as)')

    fig.set_size_inches([5,5])
    plt.tight_layout()

    def update_img(n):
        if verbose:
            print ("processing frame {0} of {1}".format(n, len(im_List)*pad_factor))
        plt_im.set_data(im_data(n))
        if mjd_range != 0:
            fig.suptitle('MJD: ' + str(im_List[int((n-n%pad_factor)//pad_factor)].mjd))
        else:
            time = im_List[int((n-n%pad_factor)//pad_factor)].time
            time_str = ("%d:%02d.%02d" % (int(time), (time*60) % 60, (time*3600) % 60))
            fig.suptitle(time_str)

        return plt_im

    ani = animation.FuncAnimation(fig,update_img,len(im_List)*pad_factor,interval=1e3/fps)
    writer = animation.writers['ffmpeg'](fps=fps, bitrate=1e6)
    ani.save(out,writer=writer,dpi=dpi)
    plt.close(fig)

def generate_observations(movie, obs_sgra, output_path='.', noise=True):
    """Generates sgra-like obeservations from the movie

    Args:
        movie (ehtim.Movie): a Movie object
        obs_sgra (observation): An Observation object with sgra observation parameters
        output_path (str): output path for caltable
        noise (bool): False for no noise

    Returns:
        obs (observation): An Observation object with sgra-like observation of the input movie.
    """
    add_th_noise = noise  # False if you *don't* want to add thermal error. If there are no sefds in obs_orig it will use the sigma for each data point
    phasecal = not noise  # True if you don't want to add atmospheric phase error. if False then it adds random phases to simulate atmosphere
    ampcal = not noise  # True if you don't want to add atmospheric amplitude error. if False then add random gain errors
    stabilize_scan_phase = noise  # if true then add a single phase error for each scan to act similar to adhoc phasing
    stabilize_scan_amp = noise  # if true then add a single gain error at each scan
    jones = noise  # apply jones matrix for including noise in the measurements (including leakage)
    inv_jones = False  # no not invert the jones matrix
    frcal = not noise  # True if you do not include effects of field rotation
    dcal = not noise  # True if you do not include the effects of leakage
    dterm_offset = 0.05  # a random offset of the D terms is given at each site with this standard deviation away from 1
    rlgaincal = not noise
    neggains = not noise

    # these gains are approximated from the EHT 2017 data
    # the standard deviation of the absolute gain of each telescope from a gain of 1
    gain_offset = {'AA': 0.15, 'AP': 0.15, 'AZ': 0.15, 'LM': 0.6, 'PV': 0.15, 'SM': 0.15, 'JC': 0.15, 'SP': 0.15,
                   'SR': 0.0}
    # the standard deviation of gain differences over the observation at each telescope
    gainp = {'AA': 0.05, 'AP': 0.05, 'AZ': 0.05, 'LM': 0.5, 'PV': 0.05, 'SM': 0.05, 'JC': 0.05, 'SP': 0.15,
             'SR': 0.0}

    obs = movie.observe_same(obs_sgra, ttype='nfft', add_th_noise=add_th_noise, ampcal=ampcal, phasecal=phasecal,
                             stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp,
                             gain_offset=gain_offset, gainp=gainp, jones=jones, inv_jones=inv_jones,
                             dcal=dcal, frcal=frcal, rlgaincal=rlgaincal, neggains=neggains,
                             dterm_offset=dterm_offset, caltable_path=output_path, sigmat=0.25)

    return obs


def load_fits(path):
    image = eh.image.load_fits(path)
    image = image.regrid_image(image.fovx(), pynoisy.noisy_core.get_image_size()[0])
    return pynoisy.envelope.grid(data=image.imarr())