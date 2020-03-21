import os
from tqdm import tqdm
from joblib import Parallel, delayed
import ehtim as eh
import pynoisy.eht_functions as ehtf
import argparse
import glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def parse_arguments():
    """Parse the command-line arguments for each run.
    The arguemnts are split into general, envelope, evolution and observations related arguments.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--eht_home',
                        default='/home/aviad/Code/eht-imaging/',
                        help='(default value: %(default)s) ehtim home directory.')
    parser.add_argument('--sgr_path',
                        default='SgrA/data/calibrated_data_oct2019/frankenstein_3599_lo_SGRA_polcal_netcal_LMTcal_10s.uvfits',
                        help='(default value: eht_home/%(default)s) Path to sgr uvfits file. Relative within the eht home directory.')
    parser.add_argument('--dir',
                        help='Directory path with hdf5 files')
    parser.add_argument('--mp4',
                        action='store_true',
                        help='Save mp4 outputs')
    parser.add_argument('--n_jobs',
                        type=int,
                        default=1,
                        help='Number of jobs to parallelize computations.')
    args = parser.parse_args()
    return args

def observe(hdf5_path, obs_sgra, args):
    """The main function to (parallel) run the parameter survey.

    Args:
        hdf_path (str) Path to hdf5 movie.
        obs_sgra (obs): SgrA observation object.
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    output_path = hdf5_path[:-5]
    movie = eh.movie.load_hdf5(hdf5_path)

    # Save mp4 for display
    if args.mp4:
        ehtf.export_movie(movie.im_list(), fps=10, out=output_path + '.mp4', verbose=False)

    obs = ehtf.generate_observations(movie, obs_sgra, output_path)
    obs.save_uvfits(output_path + '.uvfits')


if __name__ == "__main__":

    args = parse_arguments()
    obs_sgra = ehtf.load_sgra_obs(args.eht_home, args.sgr_path)
    hdf_paths = glob.glob(args.dir + '*.hdf5')
    if args.n_jobs == 1:
        for path in hdf_paths:
            observe(path, obs_sgra, args)
    else:
        Parallel(n_jobs=args.n_jobs)(delayed(observe)(path, obs_sgra, args) for path in tqdm(hdf_paths))

