import xarray as xr
import pynoisy
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
import yaml
import itertools

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default='./eigenvectors',
                        help='(default value: %(default)s) Path to output directory.')
    parser.add_argument('--config',
                        help='(default value: %(default)s) Path to config yaml file.')
    parser.add_argument('--nx',
                         type=int,
                         default=64,
                         help='(default value: %(default)s) Number of x grid points to rescale to.')
    parser.add_argument('--ny',
                         type=int,
                         default=64,
                         help='(default value: %(default)s) Number of y grid points to rescale to.')
    parser.add_argument('--nt',
                         type=int,
                         default=64,
                         help='(default value: %(default)s) Number of video frames to use.')
    parser.add_argument('--degree',
                         type=int,
                         default=24,
                         help='(default value: %(default)s) Degree of eigenvectors/modes.')
    parser.add_argument('--lobpcg_iter',
                        default=50,
                        help='(default value: %(default)s) maximum number of LOBPCG iterations.')
    parser.add_argument('--precond',
                        default=True,
                        help='(default value: %(default)s) Use SMG precodintioning.')

    args = parser.parse_args()
    return args

uniform_sample = lambda a, b: (b - a) * np.random.random_sample() + a

if __name__ == "__main__":
    # Parse input arguments
    args = parse_arguments()

    # Load input parameter file (.yaml)
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    filename = '{runname}.{yaml}.LOBPCGiter{iter}_degree{degree}_precond_{precond}_{nt}x{nx}x{ny}'.format(
        runname=config['run']['name'],
        yaml=args.config.split('/')[-1],
        iter=args.lobpcg_iter,
        degree=args.degree,
        precond=args.precond,
        nt=args.nt, nx=args.nx, ny=args.ny
    )

    hyperparams = {'nt': [args.nt], 'nx': [args.nx], 'ny': [args.ny]}
    dims_to_expand = []
    for param, value in config['params'].items():
        if (value == 'grid'):
            hyperparams[param] = np.linspace(*config['ranges'][param], config['grid'][param])
            dims_to_expand.append(param)
        elif (value == 'random'):
            hyperparams[param] = [pynoisy.utils.uniform_sample(*config['ranges'][param])]
            filename += '_{}{:1.3}'.format(param, hyperparams[param][0])
        else:
            hyperparams[param] = [value]

    # Save a NetCDF file
    if os.path.isdir(args.output_dir) is False:
        os.mkdir(args.output_dir)

    # Split parameter grid to number of files
    output = os.path.join(args.output_dir, filename)
    parameter_grid = list(itertools.product(*hyperparams.values()))
    for num_file, param_split in enumerate(tqdm(np.array_split(parameter_grid, config['run']['num_files']), desc='file')):
        output = output + '_{:03}.nc'.format(num_file) if (config['run']['num_files'] > 1) else output + '.nc'

        eigenvectors = []
        for i, parameters in enumerate(tqdm(param_split, leave=False)):
            parameter_dict = dict(zip(hyperparams.keys(), parameters))
            solver = pynoisy.forward.HGRFSolver.homogeneous(**parameter_dict)
            grf = solver.run(num_frames=int(parameter_dict['nt']), n_jobs=4, evolution_length=parameter_dict['evolution_length'], verbose=False)
            eigenvector = solver.get_eigenvectors(grf, degree=args.degree, precond=args.precond, verbose=False, maxiter=args.lobpcg_iter)
            eigenvector.coords.update(parameter_dict)
            eigenvector = eigenvector.expand_dims(dims_to_expand)
            eigenvectors.append(eigenvector)

        eigenvectors = xr.concat(eigenvectors, dim=dims_to_expand[0])
        eigenvectors.attrs = {
            'runname': config['run']['name'],
            'file_num': '{:03} / {:03}'.format(num_file, config['run']['num_files']-1),
            'num_eigenvectors': len(param_split),
            'desc': config['run']['desc'],
            'date': time.strftime("%d-%b-%Y-%H:%M:%S"),
            'lobpcg_iter': args.lobpcg_iter,
            'preconditioning': 'True' if args.precond else 'False'
        }
        eigenvectors.to_netcdf(output, mode='w')




