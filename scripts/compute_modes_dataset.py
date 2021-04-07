import numpy as np
import xarray as xr
import pynoisy
from tqdm import tqdm
import argparse, time, os, itertools, yaml

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='scripts/configs/modes.opening_angles.yaml',
                        help='(default value: %(default)s) Path to config yaml file.')

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
outpath = config['dataset']['outpath'].replace('<datetime>', time.strftime("%d-%b-%Y-%H:%M:%S"))
if os.path.isdir(os.path.dirname(outpath)) is False:
    os.mkdir(os.path.dirname(outpath))

# Generate parameter grid according to configuration file
param_grid = []
num_parameters = 0
for key, value in config.items():
    if 'variable_params' in config[key]:
        for param, grid_spec in config[key]['variable_params'].items():
            grid = np.linspace(*grid_spec['range'], grid_spec['num'])
            param_grid.append(list(zip([key]*grid_spec['num'], [param]*grid_spec['num'], grid)))
            num_parameters += 1

if num_parameters > 2:
    raise AttributeError('More than 2 parameters dataset is not supported')

param_grid = list(itertools.product(*param_grid))

# Split dataset into chunks which fit in memory
if config['dataset']['split'] > 0:
    if (grid_spec['num'] % config['dataset']['split'] != 0):
        raise AttributeError('The dataset split should be a divisible of the fastest parameter num \
                             (defined last in the config file)')
    split_list = lambda l, chunk: [l[i:i + chunk] for i in range(0, len(l), chunk)]
    param_grid = split_list(param_grid, config['dataset']['split'])
else:
    param_grid = [param_grid]

# Main iteration loop
diffusion_model = getattr(pynoisy.diffusion, config['diffusion']['model'])
advection_model = getattr(pynoisy.advection, config['advection']['model'])
advection = advection_model(config['grid']['ny'], config['grid']['nx'], **config['advection']['fixed_params'])
diffusion = diffusion_model(config['grid']['ny'], config['grid']['nx'], **config['diffusion']['fixed_params'])
solver = pynoisy.forward.HGRFSolver(advection, diffusion, config['grid']['nt'], config['grid']['evolution_length'],
                                    num_solvers=args.num_solvers)

first_iteration = True
for param_split in tqdm(param_grid, desc='parameter split', leave=True):
    modes_split = []

    for param in tqdm(param_split, desc='inner loop', leave=False):
        # Arrange according to diffusion and advection parameters
        variable_params = {'diffusion': {}, 'advection': {}}
        for p in param:
            variable_params[p[0]][p[1]] = p[2]

        # Generate a solver object and compute modes
        advection = advection_model(config['grid']['ny'], config['grid']['nx'], **config['advection']['fixed_params'], **variable_params['advection'])
        diffusion = diffusion_model(config['grid']['ny'], config['grid']['nx'], **config['diffusion']['fixed_params'], **variable_params['diffusion'])
        solver.update_advection(advection)
        solver.update_diffusion(diffusion)
        modes = pynoisy.linalg.randomized_subspace(
            solver, args.blocksize, args.maxiter, tol=args.tol, n_jobs=args.n_jobs, num_test_grfs=args.num_test_grfs,
            tqdm_bar=False)[0]

        # Expand modes dimensions to include the dataset parameters
        dims = []
        for field, params in variable_params.items():
            for key, value in params.items():
                dims.append(field + '_' + key)
                modes = modes.expand_dims({dims[-1]: [value]})
        modes_split.append(modes)

    # Concatenate and update dataset to file
    modes_split = xr.concat(modes_split, dim=dims[-1])
    modes_split.attrs.update(vars(args))
    modes_split.attrs.update(date=time.strftime("%d-%b-%Y-%H:%M:%S"))
    if first_iteration:
        modes_split.to_netcdf(outpath, unlimited_dims=dims[-2])
        first_iteration = False
    else:
        modes_split.io.append_to_netcdf(outpath, unlimited_dims=dims[-2])

