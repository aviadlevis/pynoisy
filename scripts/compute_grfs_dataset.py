"""
This scripts computes and saves a dataset of Gaussian Random Fields (GRFs) [1,2].
This scripts takes a yaml configuration file as input which describes the
dataset parameters and gridding. See `configs/opening_angles.yaml` for an example configuration file.
To see description of command line arguments and help use
    `python compute_grfs_dataset.py --h

References
----------
.. [1] Lee, D. and Gammie, C.F., 2021. Disks as Inhomogeneous, Anisotropic Gaussian Random Fields.
   The Astrophysical Journal, 906(1), p.39. url: https://iopscience.iop.org/article/10.3847/1538-4357/abc8f3/meta
.. [2] inoisy code: https://github.com/AFD-Illinois/inoisy
"""
import numpy as np
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
    parser.add_argument('--num',
                        default=10,
                        type=int,
                        help='(default value: %(default)s) Number of grfs.')

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
dirpath = os.path.join(config['dataset']['outpath'].replace('<datetime>', datetime), 'grfs')
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
    grfs = solver.run(num_samples=args.num, n_jobs=args.n_jobs, verbose=0)
    grfs.name = 'grfs'

    # Expand modes dimensions to include the dataset parameters
    for field_type, params in variable_params.items():
        for param, value in params.items():
            grfs = grfs.expand_dims({config[field_type]['variable_params'][param]['dim_name']: [value]})

    # Save to datasets
    grfs.attrs.update(attrs)
    grfs = grfs.assign_coords({'file_index': i})
    grfs.to_netcdf(os.path.join(dirpath + 'grf{:04d}.nc'.format(i)))