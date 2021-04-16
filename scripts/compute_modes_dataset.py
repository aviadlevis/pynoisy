"""
This scripts computes and saves a dataset of SPDE modes.
The modes are computed with randomized subspace iteration (RSI) [1], see pynoisy.linalg.randomized_subspace() for
details on the input arguments of RSI. This scripts takes a yaml configuration file as input which describes the
dataset parameters and gridding. See `configs/modes.opening_angles.yaml` for an example configuration file.
To see description of command line arguments and help use
    `python compute_modes_dataset.py --h

References
----------
..[1] Halko, N., Martinsson, P.G. and Tropp, J.A.. Finding structure with randomness: Probabilistic algorithms
      for constructing approximate matrix decompositions. SIAM review, 53(2), pp.217-288. 2011.
      url: https://epubs.siam.org/doi/abs/10.1137/090771806
"""

import numpy as np
import xarray as xr
import pynoisy
from tqdm import tqdm
import argparse, time, os, itertools, yaml
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

args = parse_arguments()

with open(args.config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# Setup output path and directory
datetime = time.strftime("%d-%b-%Y-%H:%M:%S")
outpath = config['dataset']['outpath'].replace('<datetime>', datetime)
dirpath = os.path.dirname(outpath)
Path(dirpath).mkdir(parents=True, exist_ok=True)

# Dataset attributes (xarray doesnt save boolean attributes)
attrs = vars(args)
for key in attrs.keys():
    attrs[key] = str(attrs[key]) if isinstance(attrs[key], bool) else attrs[key]
attrs['date'] = datetime

# Generate parameter grid according to configuration file
param_grid = []
num_parameters = 0
variable_params = dict()
for field_type, params in config.items():
    if 'variable_params' in params:
        for param, grid_spec in config[field_type]['variable_params'].items():
            grid = np.linspace(*grid_spec['range'], grid_spec['num'])
            param_grid.append(list(zip([field_type]*grid_spec['num'], [param]*grid_spec['num'], grid)))
            variable_params[field_type] = dict()
            num_parameters += 1
if num_parameters > 2:
    raise AttributeError('More than 2 parameters dataset is not supported')

param_grid = list(itertools.product(*param_grid))

# Generate solver object with the fixed parameters
diffusion_model = getattr(pynoisy.diffusion, config['diffusion']['model'])
advection_model = getattr(pynoisy.advection, config['advection']['model'])
advection = advection_model(config['grid']['ny'], config['grid']['nx'], **config['advection']['fixed_params'])
diffusion = diffusion_model(config['grid']['ny'], config['grid']['nx'], **config['diffusion']['fixed_params'])
solver = pynoisy.forward.HGRFSolver(advection, diffusion, config['grid']['nt'], config['grid']['evolution_length'],
                                    num_solvers=args.num_solvers)

# Main iteration loop: compute modes and save to dataset
for i, param in enumerate(tqdm(param_grid, desc='parameter')):
    # Arrange according to diffusion and advection parameters
    for p in param:
        variable_params[p[0]][p[1]] = p[2]

    # Generate a solver object and compute modes
    advection = advection_model(config['grid']['ny'], config['grid']['nx'], **config['advection']['fixed_params'], **variable_params['advection'])
    diffusion = diffusion_model(config['grid']['ny'], config['grid']['nx'], **config['diffusion']['fixed_params'], **variable_params['diffusion'])
    solver.update_advection(advection)
    solver.update_diffusion(diffusion)
    modes, residuals = pynoisy.linalg.randomized_subspace(solver, args.blocksize, args.maxiter, tol=args.tol,
                                               n_jobs=args.n_jobs, num_test_grfs=args.num_test_grfs, verbose=False)

    # Expand modes dimensions to include the dataset parameters
    for field_type, params in variable_params.items():
        for param, value in params.items():
            modes = modes.expand_dims({config[field_type]['variable_params'][param]['dim_name']: [value]})
            if args.save_residuals:
                residuals = residuals.expand_dims({config[field_type]['variable_params'][param]['dim_name']: [value]})

    # Save modes (and optionally residuals) to datasets
    modes.attrs.update(attrs)
    modes.to_netcdf(outpath.format(i))
    if args.save_residuals:
        residuals.to_netcdf(outpath.format(i) + '.residual')

# Consolidate residuals into a single file (and delete temporary residual files)
if args.save_residuals:
    residuals = xr.open_mfdataset(os.path.join(dirpath, '*residual'))
    residuals.attrs.update(attrs)
    residuals.to_netcdf(os.path.join(dirpath, 'residuals.nc'))
    for p in Path(dirpath).glob('*.residual'):
        p.unlink()