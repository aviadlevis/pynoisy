import pynoisy
import numpy as np
import os, time
from tqdm import tqdm
from joblib import Parallel, delayed
import ehtim.imaging.dynamical_imaging as di
import pynoisy.eht_functions as ehtf
import itertools
import argparse
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def generate_noisy_movie(envelope_params, evolution_params, verbose):
    """Generate a noisy movie using the pynoisy library

    Returns:
        movie (Movie): A Movie object containing the movie frames.
    """
    rotation = pynoisy.RotationDirection.clockwise if evolution_params['rotation'] == 'cw' else \
        pynoisy.RotationDirection.counter_clockwise
    envelope = pynoisy.RingEnvelope(
        inner_radius=envelope_params['inner_radius'],
        bright_ring_thickness=envelope_params['bright_ring_thickness'],
        bright_ring_decay=envelope_params['bright_ring_decay'],
        ascent=envelope_params['ascent'],
        inner_decay=envelope_params['inner_decay'],
        amplitude=evolution_params['amp'],
    )
    advection = pynoisy.DiskAdvection(direction=rotation)
    diffusion = pynoisy.RingDiffusion(opening_angle=-evolution_params['angle']*rotation.value)
    solver = pynoisy.PDESolver(advection, diffusion, envelope, evolution_params['eps'])
    movie = solver.run(evolution_params['dur'], verbose=verbose)
    return movie

def main(envelope_params, evolution_params, obs_sgra, args):
    noisy_movie = generate_noisy_movie(envelope_params, evolution_params, args.verbose)
    ehtim_movie = ehtf.ehtim_movie(noisy_movie.frames, obs_sgra, envelope_params['flux'], envelope_params['fov'])

    # Save Noisy and ehtim Movie objects
    output_folder = os.path.join(args.eht_home, args.output_folder, time.strftime("%d-%b-%Y-%H:%M:%S"))

    output_path = os.path.join(
        output_folder,
        'envelope' + ''.join(['_{}{}'.format(key, value) for key, value in envelope_params.items()]) +
        '_evolution' + ''.join(['_{}{}'.format(key, value) for key, value in evolution_params.items()])
    )
    noisy_movie.save(output_path + '.pkl')
    ehtim_movie.save_hdf5(output_path + '.hdf5')

    # Export arguments as json
    with open('args.txt', 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    # Save an mp4 for display
    di.export_movie(ehtim_movie.im_list(), fps=10, out=output_path + '.mp4')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eht_home',
                        default='/home/aviad/Code/eht-imaging/',
                        help='(default value: %(default)s) ehtim home directory.')
    parser.add_argument('--sgr_path',
                        default='SgrA/data/calibrated_data_oct2019/frankenstein_3599_lo_SGRA_polcal_netcal_LMTcal_10s.uvfits',
                        help='(default value: eht_home/%(default)s) Path to sgr uvfits file. Relative within the eht home directory.')
    parser.add_argument('--output_folder',
                        default='SgrA/data/synthetic_rings/',
                        help='Path to a json input argument file.')
    parser.add_argument('--args_path',
                        help='Path to a json input argument file.')
    parser.add_argument('--n_jobs',
                        type=int,
                        default=1,
                        help='Number of jobs to parallelize computations.')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='True to print out noisy iteration ouputs')
    # Envelope parameters
    envelope = parser.add_argument_group('Envelope parameters')
    envelope.add_argument('--flux',
                        nargs='+',
                        type=np.float32,
                        default=[2.23],
                        help='(default value: %(default)s) Normalize the average flux of the movie.')
    envelope.add_argument('--fov',
                        nargs='+',
                        type=np.float32,
                        default=[95.0],
                        help='(default value: %(default)s) Field of view of the image micro-arcsecs.')
    envelope.add_argument('--inner_radius',
                        nargs='+',
                        type=np.float32,
                        default=[0.2],
                        help='(default value: %(default)s) Inner ring radius (unitless).')
    envelope.add_argument('--bright_ring_thickness',
                        nargs='+',
                        type=np.float32,
                        default=[0.05],
                        help='(default value: %(default)s) Thickness of the bright ring (unitless).')
    envelope.add_argument('--bright_ring_decay',
                        nargs='+',
                        type=np.float32,
                        default=[100],
                        help='(default value: %(default)s) Decay constant of the bright ring exponential.')
    envelope.add_argument('--inner_decay',
                        nargs='+',
                        type=np.float32,
                        default=[5.0],
                        help='(default value: %(default)s) Decay constant of the exponential.')
    envelope.add_argument('--ascent',
                        nargs='+',
                        type=np.float32,
                        default=[1.0],
                        help='(default value: %(default)s) Ascent of the inner (dark) region.')

    # Evolution parameters
    evolution = parser.add_argument_group('Evolution parameters')
    evolution.add_argument('--angle',
                        nargs='+',
                        type=np.float32,
                        default=[np.pi/3.0],
                        help='(default value: %(default)s) Opening angle of the diffusion tensor '
                             'with respect to local radius.')
    evolution.add_argument('--rotation',
                        nargs='+',
                        choices=['cw', 'ccw'],
                        default=['cw'],
                        help='(default value: %(default)s) Rotation directions. cw: clockwise. ccw: counter clockwise.')
    evolution.add_argument('--eps',
                        nargs='+',
                        type=np.float32,
                        default=[0.1],
                        help='(default value: %(default)s) Forcing strength.')
    evolution.add_argument('--dur',
                        nargs='+',
                        type=np.float32,
                        default=[0.1],
                        help='(default value: %(default)s) Evolution duration.')
    evolution.add_argument('--amp',
                        nargs='+',
                        type=np.float32,
                        default=[0.2],
                        help='(default value: %(default)s) Noisy amplitude '
                             'output_image = envelope * exp(amp * noisy_image).')


    args = parser.parse_args()

    # Load input arguments
    if args.args_path is not None:
        with open(args.args_path, 'r') as file:
            args.__dict__ = json.load(file)

    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)
    envelope_args = arg_groups['Envelope parameters'].__dict__
    evolution_args = arg_groups['Evolution parameters'].__dict__

    return args, envelope_args, evolution_args

if __name__ == "__main__":

    args, envelope_args, evolution_args = parse_arguments()

    obs_sgra = ehtf.load_sgra_obs(args.eht_home, args.sgr_path)

    envelope_params = itertools.product(*envelope_args.values())
    evolution_params = itertools.product(*evolution_args.values())
    parameters = itertools.product(envelope_params, evolution_params)

    if args.n_jobs == 1:
        for params in parameters:
            main(dict(zip(envelope_args.keys(), params[0])), dict(zip(evolution_args.keys(), params[1])), obs_sgra, args)
    else:
        Parallel(n_jobs=args.n_jobs)(delayed(main)(
            dict(zip(envelope_args.keys(), params[0])), dict(zip(evolution_args.keys(), params[1])),
            obs_sgra, args) for params in tqdm(parameters))

