import pynoisy
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import ehtim.imaging.dynamical_imaging as di
import pynoisy.eht_functions as ehtf
import itertools

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def generate_noisy_movie(opening_angle, direction1, direction2, forcing_strength, evolution_length, envelope_amplitude, source_ratios):
    """Generate a noisy movie using the pynoisy library

    Args:
       TODO

    Returns:
        movie (Movie): A Movie object containing the movie frames.
    """
    angle = np.deg2rad(105.89)
    distance = 55.56 / 160.0
    radius1 = 32.22 / 320.0 if source_ratios['size'] == 1 else 36.67 / 320.0
    radius2 = radius1 * source_ratios['size']
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), -np.cos(angle)]]).T

    # Define envelope
    disk1 = pynoisy.DiskEnvelope(radius=radius1, decay=10, amplitude=envelope_amplitude)
    disk1.shift(*np.dot(rotation, np.array([distance / 2.0, 0.0])))
    disk2 = pynoisy.DiskEnvelope(radius=radius2, decay=10, amplitude=envelope_amplitude)
    disk2.shift(*np.dot(rotation, np.array([-distance / 2.0, 0.0])))
    envelope = (disk1 + disk2 * source_ratios['flux']) / (1.0 + source_ratios['flux'])

    # Define advection
    advection1 = pynoisy.DiskAdvection(scaling_radius=radius1, direction=direction1)
    advection1.shift(*np.dot(rotation, np.array([distance / 2.0, 0.0])))
    advection2 = pynoisy.DiskAdvection(scaling_radius=radius2, direction=direction2)
    advection2.shift(*np.dot(rotation, np.array([-distance / 2.0, 0.0])))
    advection = advection1 + advection2

    # Define diffusion
    diffusion1 = pynoisy.DiskDiffusion(scaling_radius=radius1)
    diffusion2 = pynoisy.DiskDiffusion(scaling_radius=radius2)
    indices = np.bitwise_and(diffusion1.r >= radius1*0.8, diffusion1.r <= radius1 * 1.2)
    diffusion1[indices] = pynoisy.RingDiffusion(opening_angle=-opening_angle * direction1.value)
    indices = np.bitwise_and(diffusion2.r >= radius2 * 0.8, diffusion2.r <= radius2 * 1.2)
    diffusion2[indices] = pynoisy.RingDiffusion(opening_angle=-opening_angle * direction2.value)
    diffusion1.shift(*np.dot(rotation, np.array([distance / 2.0, 0.0])))
    diffusion2.shift(*np.dot(rotation, np.array([-distance / 2.0, 0.0])))
    diffusion = diffusion1 + diffusion2

    solver = pynoisy.PDESolver(advection, diffusion, envelope, forcing_strength)
    movie = solver.run(evolution_length, verbose=False)
    return movie

def main(params, total_flux, fov, obs_sgra, ehtim_home):
    """

    :param hyperparams:
    :param obs_sgra:
    :return:
    """
    noisy_movie = generate_noisy_movie(*params)
    ehtim_movie = ehtf.ehtim_movie(noisy_movie.frames, obs_sgra, total_flux, fov)

    # Save Noisy and ehtim Movie objects
    if params[6]['size'] == 1.0:
        output_path = os.path.join(ehtim_home,
                                   'SgrA/synthetic_dbsrc_symmetric/angle{:1.2f}_dir1_{}_dir2_{}_eps{:1.2f}_len{:1.1f}_amp{:1.2f}'.
                                   format(params[0], params[1].name, params[2].name, params[3], params[4], params[5]))
    else:
        output_path = os.path.join(ehtim_home,
                                   'SgrA/synthetic_dbsrc_asymmetric/angle{:1.2f}_dir1_{}_dir2_{}_eps{:1.2f}_len{:1.1f}_amp{:1.2f}'.
                                   format(params[0], params[1].name, params[2].name, params[3], params[4], params[5]))
    noisy_movie.save(output_path + '.pkl')
    ehtim_movie.save_hdf5(output_path + '.hdf5')

    # Save a gif/mp4 for display
    im_list = [ehtim_movie.get_frame(i) for i in range(ehtim_movie.nframes)]
    di.export_movie(im_list, fps=10, out=output_path + '.mp4')

if __name__ == "__main__":
    ehtim_home = '/home/aviad/Code/eht-imaging/'
    sgr_uvfits_path = 'SgrA/data/calibrated_data_oct2019/frankenstein_3599_lo_SGRA_polcal_netcal_LMTcal_10s.uvfits'

    # Hyper-parameters for ring structure
    obs_sgra = ehtf.load_sgra_obs(ehtim_home, sgr_uvfits_path)
    total_flux = 2.23
    fov = 160

    # Parameters for different ring evolutions
    opening_angles = [np.pi / 3, np.pi / 4]
    rotation_directions1 = [pynoisy.RotationDirection.clockwise, pynoisy.RotationDirection.counter_clockwise]
    rotation_directions2 = [pynoisy.RotationDirection.clockwise, pynoisy.RotationDirection.counter_clockwise]
    forcing_strengths = [0.01, 0.05, 0.1]
    evolution_lengths = [0.1, 0.5]
    envelope_amplitudes = [0.05, 0.1, 0.2]
    source_ratios = [{'flux': 1.0, 'size': 1.0},
                     {'flux': 0.6, 'size': 1.32}]

    parameters = itertools.product(opening_angles, rotation_directions1, rotation_directions2, forcing_strengths, evolution_lengths, envelope_amplitudes, source_ratios)
    Parallel(n_jobs=24)(delayed(main)(params, total_flux, fov, obs_sgra, ehtim_home) for params in tqdm(parameters))

