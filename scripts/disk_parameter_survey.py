import pynoisy
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import ehtim.imaging.dynamical_imaging as di
import pynoisy.eht_functions as ehtf
import itertools

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def generate_noisy_movie(opening_angle, rotation_direction, forcing_strength, evolution_length, envelope_amplitude, tensor_ratio, decay):
    """Generate a noisy movie using the pynoisy library

    Args:
        opening_angle (float): The diffusion tensor opening angle with respect to the local radius
                               (pi/2 means same direction of radius)
        rotation_direction (RotationDirection): Clockwise or counter-clockwise
        forcing_strength (float): Strength of the noise source term.
        evolution_length (float): Length of the evolution (determines the evolution "speed").
        envelope_amplitude (float): Amplitude of the envelope. Images are output as:
                                    Envelope * exp(-Amplitude * del) where del is the random field
        output_path (str or None): If output_path is None the movie wont be saved.

    Returns:
        movie (Movie): A Movie object containing the movie frames.
    """
    envelope = pynoisy.DiskEnvelope(radius=39.346734 / 160.0, decay=decay, amplitude=envelope_amplitude)
    advection = pynoisy.DiskAdvection(direction=rotation_direction)
    radius = 0.75 * 39.346734 / 160.0
    diffusion = pynoisy.RingDiffusion(opening_angle=-rotation_direction.value * np.pi / 2)
    diffusion._diffusion_coefficient += 0.1
    diffusion[diffusion.r >= radius] = pynoisy.RingDiffusion(opening_angle=-opening_angle*rotation_direction.value)
    solver = pynoisy.PDESolver(advection, diffusion, envelope, forcing_strength)
    movie = solver.run(evolution_length, verbose=True)
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
    output_path = os.path.join(ehtim_home, 'SgrA/synthetic_disks/TEST_angle{:1.1f}_{}_eps{:1.2f}_len{:1.1f}_amp{:1.2f}_rat{:1.1f}_dec{:2.0f}'.
                               format(params[0], params[1].name, params[2], params[3], params[4], params[5], params[6]))
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
    # opening_angles = [np.pi/3.0, np.pi/4]
    # rotation_directions = [pynoisy.RotationDirection.clockwise, pynoisy.RotationDirection.counter_clockwise]
    # forcing_strengths = [0.01, 0.025]
    # evolution_lengths = [0.1, 0.5]
    # envelope_amplitudes = [0.1]
    # tensor_ratios = [0.1]
    # decays = [10, 20]

    # parameters = itertools.product(opening_angles, rotation_directions, forcing_strengths, evolution_lengths, envelope_amplitudes, tensor_ratios, decays)
    # Parallel(n_jobs=24)(delayed(main)(params, total_flux, fov, obs_sgra, ehtim_home) for params in tqdm(parameters))
    params = [np.pi/3.0, pynoisy.RotationDirection.counter_clockwise, 0.05, 0.1, 0.15, 0.3, 5]
    main(params, total_flux, fov, obs_sgra, ehtim_home)

