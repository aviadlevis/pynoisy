"""
This scripts computes and saves a dataset of SPDE modes.
The modes are computed with randomized subspace iteration (RSI) [1], see pynoisy.linalg.randomized_subspace() for
details on the input arguments of RSI. This scripts takes a yaml configuration file as input which describes the
dataset parameters and gridding. See `configs/opening_angles.yaml` for an example configuration file.
To see description of command line arguments and help use
    `python compute_modes_dataset.py --h

References
----------
..[1] Halko, N., Martinsson, P.G. and Tropp, J.A.. Finding structure with randomness: Probabilistic algorithms
      for constructing approximate matrix decompositions. SIAM review, 53(2), pp.217-288. 2011.
      url: https://epubs.siam.org/doi/abs/10.1137/090771806
"""
import xarray as xr
import pynoisy
import pynoisy.script_utils as script_utils
from tqdm import tqdm
import argparse, time, os
import shutil, ruamel_yaml
from pathlib import Path

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.')
    parser.add_argument('--save_residuals',
                        action='store_true',
                        help='Save modes residuals to files with the same format as modes.')

    # Randomized subspace iteration (RSI) parameters
    parser.add_argument('--blocksize',
                        default=50,
                        type=int,
                        help='(default value: %(default)s) Block size for random subspace iteration (RSI).')
    parser.add_argument('--maxiter',
                        default=20,
                        type=int,
                        help='(default value: %(default)s) Maximum number of RSI iterations.')
    parser.add_argument('--tol',
                        default=1e-3,
                        type=float,
                        help='(default value: %(default)s) Tolerance (slope) for convergence'
                             'residual statistics. Used if num_test_grfs > 0.')
    parser.add_argument('--num_test_grfs',
                        default=10,
                        type=int,
                        help='(default value: %(default)s) Number of test GRFs used to gather residual statistics.'
                              'If set to zero than adaptive convergence criteria is ignored.')

    # Parallel processing parameters
    parser.add_argument('--n_jobs',
                         type=int,
                         default=4,
                         help='(default value: %(default)s) Number of jobs for MPI processing of HYPRE.')
    parser.add_argument('--num_solvers',
                         type=int,
                         default=5,
                         help='(default value: %(default)s) Number of solvers used for RSI parallel processing.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    with open(args.config, 'r') as stream:
        config = ruamel_yaml.load(stream, Loader=ruamel_yaml.Loader)

    # Datetime and github version information
    datetime = time.strftime("%d-%b-%Y-%H:%M:%S")
    github_version = pynoisy.utils.github_version()

    # Setup output path and directory
    dirpath = os.path.join(config['dataset']['outpath'].replace('<datetime>', datetime), 'modes')
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    # Setup NetCDF attributes
    attrs = script_utils.netcdf_attrs(vars(args))

    # Generate parameter grid according to configuration file
    param_grid = script_utils.get_parameter_grid(config)

    # Generate solver object with the fixed parameters
    solver = script_utils.get_default_solver(config)

    # Generate solver object with the fixed parameters
    solver = script_utils.get_default_solver(config, variable_params={'solver': {'num_solvers': args.num_solvers}})

    # Main iteration loop
    for i, params in enumerate(tqdm(param_grid, desc='parameter')):

        # Update solver object according to the variable parameters
        solver = script_utils.get_default_solver(config, params)
        modes, residuals = pynoisy.linalg.randomized_subspace(solver, args.blocksize, args.maxiter, tol=args.tol,
                                                              n_jobs=args.n_jobs, num_test_grfs=args.num_test_grfs,
                                                              verbose=False)

        # Expand modes dimensions to include the dataset parameters
        modes = script_utils.expand_dataset_dims(modes, config, params)
        residuals = script_utils.expand_dataset_dims(residuals, config, params)

        # Save modes (and optionally residuals) to datasets
        modes.attrs.update(attrs)
        modes = modes.assign_coords({'file_index': i})
        modes.to_netcdf(os.path.join(dirpath, 'mode{:04d}.nc'.format(i)))
        if args.save_residuals:
            residuals.to_netcdf(os.path.join(dirpath, 'residual{:04d}.nc'.format(i)))

    # Consolidate residuals into a single file (and delete temporary residual files)
    if args.save_residuals:
        with xr.open_mfdataset(os.path.join(dirpath, 'residual*.nc')) as dataset:
            dataset.attrs.update(attrs)
            dataset.to_netcdf(os.path.join(dirpath, 'residuals.nc'))
        for p in Path(dirpath).glob('residual*'):
            p.unlink()

    # Copy script and config for reproducibility
    shutil.copy(__file__, os.path.join(dirpath, 'script.py'.format(datetime)))
    shutil.copy(args.config, os.path.join(dirpath, 'config.yaml'.format(datetime)))