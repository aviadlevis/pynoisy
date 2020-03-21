import pynoisy
import numpy as np
import os, time
from tqdm import tqdm
from joblib import Parallel, delayed
import ehtim.scattering as so
import ehtim as eh
import pynoisy.eht_functions as ehtf
import itertools
import argparse
import json
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def parse_arguments():
    """Parse the command-line arguments for each run.
    The arguemnts are split into general, envelope, evolution and observations related arguments.

    Returns:
        parser (argparse.parser): the argument parser.
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--eht_home',
                        default='/home/aviad/Code/eht-imaging/',
                        help='(default value: %(default)s) ehtim home directory.')
    parser.add_argument('--sgr_path',
                        default='SgrA/data/calibrated_data_oct2019/frankenstein_3599_lo_SGRA_polcal_netcal_LMTcal_10s.uvfits',
                        help='(default value: eht_home/%(default)s) Path to sgr uvfits file. Relative within the eht home directory.')
    parser.add_argument('--output_folder',
                        help='Relative within the fits_path directory. If none is given, the current date-time is taken as the output folder.')
    parser.add_argument('--args_path',
                        help='Path to a json input argument file.')
    parser.add_argument('--n_jobs',
                        type=int,
                        default=1,
                        help='Number of jobs to parallelize computations.')
    parser.add_argument('--resume_run',
                        default=None,
                        help='Path to a folder to resume run.')
    parser.add_argument('--pkl',
                        action='store_true',
                        help='Save pkl Noisy outputs')
    parser.add_argument('--uvfits',
                        action='store_true',
                        help='Save uvfits observation outputs')
    parser.add_argument('--scatter',
                        action='store_true',
                        help='Save scattered outputs')
    parser.add_argument('--mp4',
                        action='store_true',
                        help='Save mp4 outputs')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='True to print out noisy iteration outputs')

    # Envelope parameters
    envelope = parser.add_argument_group('Envelope parameters')
    envelope.add_argument('--fits_path',
                          default='SgrA/synthetic_data_SGRA_3599_lo/representative_models',
                          help='(default value: eht_home/%(default)s) Paths to envelope fits files. Relative within the eht home directory.')

    # Evolution parameters
    evolution = parser.add_argument_group('Evolution parameters')
    evolution.add_argument('--angle',
                        nargs='+',
                        type=float,
                        default=[60.0],
                        help='(default value: %(default)s) Opening angle of the diffusion tensor '
                             'with respect to local radius.')
    evolution.add_argument('--radius',
                        nargs='+',
                        type=float,
                        default=[19.0],
                        help='(default value: %(default)s) Scaling radius for the diffusion/advection fields.')
    evolution.add_argument('--rotation',
                        nargs='+',
                        choices=['cw', 'ccw'],
                        default=['cw'],
                        help='(default value: %(default)s) Rotation directions. cw: clockwise. ccw: counter clockwise.')
    evolution.add_argument('--eps',
                        nargs='+',
                        type=float,
                        default=[0.1],
                        help='(default value: %(default)s) Forcing strength.')
    evolution.add_argument('--dur',
                        nargs='+',
                        type=float,
                        default=[0.1],
                        help='(default value: %(default)s) Evolution duration.')
    evolution.add_argument('--amp',
                        nargs='+',
                        type=float,
                        default=[0.2],
                        help='(default value: %(default)s) Noisy amplitude '
                             'output_image = envelope * exp(amp * noisy_image).')
    evolution.add_argument('--tau',
                        nargs='+',
                        type=float,
                        default=[1.0],
                        help='(default value: %(default)s) product of correlation time and local Keplerian frequency.')
    evolution.add_argument('--lam',
                        nargs='+',
                        type=float,
                        default=[0.5],
                        help='(default value: %(default)s) Ratio of correlation length to local radius.')
    evolution.add_argument('--tensor_ratio',
                        nargs='+',
                        type=float,
                        default=[0.1],
                        help='(default value: %(default)s) Ratio for the diffusion tensor along the two axis.')

    # Observation parameters
    observation = parser.add_argument_group('Observation parameters')
    observation.add_argument('--day',
                             nargs='+',
                             type=int,
                             default=[3599],
                             help='(default value: %(default)s) Linear polarization coherence length (uas).')
    observation.add_argument('--lpolmag',
                             nargs='+',
                             type=float,
                             default=[0.3],
                             help='(default value: %(default)s) Linear polarization fraction.')
    observation.add_argument('--lpolcorr',
                             nargs='+',
                             type=float,
                             default=[10.0],
                             help='(default value: %(default)s) Linear polarization coherence length (uas).')
    observation.add_argument('--cpolmag',
                             nargs='+',
                             type=float,
                             default=[0.1],
                             help='(default value: %(default)s) Circular polarization fraction.')
    observation.add_argument('--cpolcorr',
                             nargs='+',
                             type=float,
                             default=[5.0],
                             help='(default value: %(default)s) Circular polarization coherence length (uas).')

    args = parser.parse_args()

    args.output_folder = time.strftime("%d-%b-%Y-%H:%M:%S") if args.output_folder is None else args.output_folder
    return parser, args

def generate_noisy_movie(envelope_params, evolution_params, output_path, verbose):
    """Generates and a noisy movie using the pynoisy library.
    A pkl file is saved for the noisy movie and an HDF5 for the ehtim movie.

    Args:
        envelope_params (dict): dictionary with all the enevelope parameters.
        envelope_params (dict): dictionary with all the evolution parameters.
        verbose (bool): True for noisy iteration print out.

    Returns:
        movie (pynoisy.Movie): A movie object containing the frames
        fov (float): the field of view in radians
    """
    image = eh.image.load_fits(envelope_params['envelope'])
    image = image.regrid_image(image.fovx(), pynoisy.core.get_image_size()[0])
    fov = image.fovx()

    if os.path.exists(output_path + '.pkl') == False:
        rotation = pynoisy.RotationDirection.clockwise if evolution_params['rotation'] == 'cw' else \
            pynoisy.RotationDirection.counter_clockwise
        unitless_radius = eh.RADPERUAS * evolution_params['radius'] / fov
        envelope = pynoisy.Envelope(data=image.imarr(), amplitude=evolution_params['amp'])
        diffusion = pynoisy.RingDiffusion(
            opening_angle=-np.deg2rad(evolution_params['angle'])*rotation.value,
            tau=evolution_params['tau'],
            lam=evolution_params['lam'],
            scaling_radius=unitless_radius,
            tensor_ratio=evolution_params['tensor_ratio']
        )
        advection = pynoisy.DiskAdvection(direction=rotation, scaling_radius=unitless_radius)
        solver = pynoisy.PDESolver(advection, diffusion, envelope, forcing_strength=evolution_params['eps'])
        movie = solver.run(evolution_length=evolution_params['dur'], verbose=verbose)
    else:
        movie = pynoisy.Movie()
        movie.load(output_path + '.pkl')
    return movie, fov


def main(envelope_params, evolution_params, observation_params, obs_sgra, args, output_folder):
    """The main function to (parallel) run the parameter survey.

    Args:
        envelope_params (dict): dictionary with all the enevelope parameters.
        envelope_params (dict): dictionary with all the evolution parameters.
        obs_sgra (obs): SgrA observation object.
        args (Namespace): an argparse Namspace object with all the command line arguments.
        output_folder (str): the output directory where all results are saved.
    """
    round_digits = lambda x: round(x, 3) if isinstance(x, float) else x
    output_path = os.path.join(
        output_folder, envelope_params['envelope'].split('/')[-1][:-5] +
        ''.join(['_{}{}'.format(key, round_digits(value)) for key, value in evolution_params.items()]))

    noisy_movie, fov = generate_noisy_movie(envelope_params, evolution_params, output_path, args.verbose)
    if args.pkl:
        noisy_movie.save(output_path + '.pkl')

    if os.path.exists(output_path + '.hdf5') == False:
        ehtim_movie = ehtf.ehtim_movie(noisy_movie.frames, obs_sgra, normalize_flux=False, fov=fov, fov_units='rad',
                                       start_time=obs_sgra.tstart - 0.08,
                                       linpol_mag=observation_params['lpolmag'],
                                       linpol_corr=observation_params['lpolcorr'],
                                       circpol_mag=observation_params['cpolmag'],
                                       cirpol_corr=observation_params['cpolcorr'])
        ehtim_movie.save_hdf5(output_path + '.hdf5')
    else:
        ehtim_movie = eh.movie.load_hdf5(output_path + '.hdf5')

    # Save mp4 for display
    if args.mp4 and os.path.exists(output_path + '.mp4') == False:
        ehtf.export_movie(ehtim_movie.im_list(), fps=10, out=output_path + '.mp4', verbose=False)

    # Generate measurements
    if args.scatter and os.path.exists(output_path + '_scattered.hdf5') == False:
        eps = so.MakeEpsilonScreen(ehtim_movie.xdim, ehtim_movie.ydim, rngseed=34)
        scattering_model = so.ScatteringModel()
        ehtim_movie = scattering_model.Scatter_Movie(ehtim_movie, eps)
        ehtim_movie.save_hdf5(output_path + '_scattered.hdf5')
        if args.mp4 and os.path.exists(output_path + '_scattered.mp4') == False:
            ehtf.export_movie(ehtim_movie.im_list(), fps=10, out=output_path + '_scattered.mp4', verbose=False)

    output_path += ''.join(['_{}{}'.format(key, round_digits(value)) for key, value in observation_params.items()])
    if args.uvfits and os.path.exists(output_path + '.uvfits') == False:
        obs = ehtf.generate_observations(ehtim_movie, obs_sgra, output_path)
        obs.save_uvfits(output_path + '.uvfits')

def split_arguments(parser, args):
    """Split arguments into envelope and evolution related arguments.

    Args:
        parser (argparse.parser): the argument parser.
        args (Namespace): an argparse Namspace object with all the command line arguments.

    Returns:
        envelope_args (dict): dictionary with all the enevelope parameters.
        evolution_args (dict): dictionary with all the evolution parameters.
        observation_args (dict): dictionary with all the observation parameters.
    """
    arg_groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = argparse.Namespace(**group_dict)

    envelope_args = arg_groups['Envelope parameters'].__dict__
    envelope_args['envelope'] = glob.glob(os.path.join(args.eht_home, envelope_args.pop('fits_path') + '/*.fits'))
    if not envelope_args['envelope']:
        raise AttributeError('No fits envelopes found at folder: {}'.format(os.path.join(args.eht_home, args.fits_path)))

    evolution_args = arg_groups['Evolution parameters'].__dict__
    observation_args = arg_groups['Observation parameters'].__dict__
    return envelope_args, evolution_args, observation_args

def load_save_parameters(parser, args):
    """Load and save run arguments or resume previous run.

    Args:
        parser (argparse.parser): the argument parser.
        args (Namespace): an argparse Namspace object with all the command line arguments.

    Returns:
        envelope_args (dict): dictionary with all the enevelope parameters.
        evolution_args (dict): dictionary with all the evolution parameters.
        observation_args (dict): dictionary with all the observation parameters.
    """
    if args.resume_run is not None:
        output_folder = os.path.join(args.eht_home, args.resume_run)
        args.args_path = os.path.join(output_folder, 'args.txt')
    else:
        output_folder = os.path.join(args.eht_home, args.fits_path, args.output_folder)

    if args.args_path is not None:
        with open(args.args_path, 'r') as file:
            args.__dict__ = {**args.__dict__, **json.load(file)}
    envelope_args, evolution_args, observation_args = split_arguments(parser, args)

    # Export arguments as json
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, 'args.txt'), 'w') as file:
        json.dump({**envelope_args, **evolution_args, **observation_args}, file, indent=2)
    return envelope_args, evolution_args, observation_args, output_folder

if __name__ == "__main__":

    parser, args = parse_arguments()

    envelope_args, evolution_args, observation_args, output_folder = load_save_parameters(parser, args)

    envelope_params = itertools.product(*envelope_args.values())
    evolution_params = itertools.product(*evolution_args.values())
    observation_params = itertools.product(*observation_args.values())
    parameters = itertools.product(envelope_params, evolution_params, observation_params)

    obs_sgra = ehtf.load_sgra_obs(args.eht_home, args.sgr_path)

    if args.n_jobs == 1:
        for params in parameters:
            main(dict(zip(envelope_args.keys(), params[0])),
                 dict(zip(evolution_args.keys(), params[1])),
                 dict(zip(observation_args.keys(), params[2])),
                 obs_sgra, args, output_folder)
    else:
        Parallel(n_jobs=args.n_jobs)(delayed(main)(
            dict(zip(envelope_args.keys(), params[0])),
            dict(zip(evolution_args.keys(), params[1])),
            dict(zip(observation_args.keys(), params[2])),
            obs_sgra, args, output_folder) for params in tqdm(parameters))

