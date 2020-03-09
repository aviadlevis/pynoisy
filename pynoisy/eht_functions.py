import ehtim as eh
import numpy as np
import os

def load_sgra_obs(ehtim_home, uvfits_path):
    """Load SgrA observations.

    Args:
        eht_home (str):  Directory where eht-imaging library is located.
        uvfits_path (str): Relative path from the eht_home directory to the observation file.

    Returns:
        obs_sgra (Obsdata): observation object
    """
    obsfilename = os.path.join(ehtim_home, uvfits_path)
    obs_sgra = eh.obsdata.load_uvfits(obsfilename, remove_nan=True)

    # Load telescope site locations and SEFDs (how noisy they are)
    eht = eh.array.load_txt(os.path.join(ehtim_home, 'arrays/EHT2017_m87.txt'))

    # Copy the correct mount types
    t_obs = list(obs_sgra.tarr['site'])
    t_eht = list(eht.tarr['site'])
    t_conv = {'AA': 'ALMA', 'AP': 'APEX', 'SM': 'SMA', 'JC': 'JCMT', 'AZ': 'SMT', 'LM': 'LMT', 'PV': 'PV', 'SP': 'SPT'}
    for t in t_conv.keys():
        if t in obs_sgra.tarr['site']:
            for key in ['fr_par', 'fr_elev', 'fr_off']:
                obs_sgra.tarr[key][t_obs.index(t)] = eht.tarr[key][t_eht.index(t_conv[t])]

    return obs_sgra

def ehtim_movie(frames, obs_sgra, total_flux=2.23, fov=95, start_time=None, end_time=None):
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
    fov *= eh.RADPERUAS

    start_time = obs_sgra.tstart if start_time is None else start_time
    end_time = obs_sgra.tstop if end_time is None else end_time
    times = np.linspace(start_time, end_time, len(frames))

    movie_frames = []
    flux_normalization = total_flux / frames.mean(axis=0).sum()
    for frame, time in zip(frames, times):
        im = eh.image.make_empty(frame.shape[0], fov, ra, dec, rf, source='SgrA')
        im.mjd = mjd
        im.time = time
        im.imvec = flux_normalization * frame.reshape(-1)
        movie_frames.append(im)

    movie = eh.movie.merge_im_list(movie_frames)

    # change the synthetic image coordinates to align with the obs
    movie.ra = obs_sgra.ra
    movie.dec = obs_sgra.dec
    movie.rf = obs_sgra.rf
    movie.reset_interp(interp='linear', bounds_error=False)
    return movie

