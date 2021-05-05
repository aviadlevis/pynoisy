"""
This scripts benchmarks the Gaussian Random Field (GRF) loss manifold across a grid of different true parameters.
The grid is defined by the benchmark.yaml configuration file.
"""
import numpy as np
import xarray as xr
import pynoisy
import pynoisy.script_utils as script_utils
from tqdm import tqdm
import argparse, time, os, ruamel_yaml
from pathlib import Path
import shutil

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.')
    parser.add_argument('--degree',
                        type=int,
                        help='(default value: %(default)s) Modes degree. If None or larger than the modes dataset \
                               then the maximum degree is used.')
    parser.add_argument('--damp',
                        default=0.0,
                        type=float,
                        help='(default value: %(default)s) Dampning factor.')
    parser.add_argument('--num_samples',
                        default=1,
                        type=int,
                        help='(default value: %(default)s) Number of random samples.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    with open(args.config, 'r') as stream:
        config = ruamel_yaml.load(stream, Loader=ruamel_yaml.Loader)

    # Datetime and github version information
    datetime = time.strftime("%d-%b-%Y-%H:%M:%S")
    github_version = pynoisy.utils.github_version()

    # Setup paths
    dirpath = config['output']['path']
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    mode_directory = config['modes']['directory']

    # Setup NetCDF attributes
    attrs = script_utils.netcdf_attrs(vars(args))

    # Generate parameter grid according to configuration file
    param_grid = script_utils.get_parameter_grid(config)

    # Generate solver object with the fixed parameters
    solver = script_utils.get_default_solver(config)

    # Load modes and generate subspace
    modes = xr.open_mfdataset(os.path.join(mode_directory, 'mode*.nc'))
    modes = modes.isel(degree=slice(args.degree)) if args.degree is not None else modes
    attrs.update(degree=modes.degree.size)
    subspace = modes.eigenvectors * modes.eigenvalues

    # Fix seeds across parameters
    seeds = [np.random.randint(0, 32767) for i in range(args.num_samples)]

    # Main iteration loop
    for i, params in enumerate(tqdm(param_grid, desc='parameter')):

        # Update solver object according to the variable parameters
        solver = script_utils.get_default_solver(config, params)

        # Compute losses as a function of seed
        losses = []
        for seed in seeds:
            solver.reseed(seed=seed, printval=False)
            grf = solver.run(n_jobs=4, verbose=0)
            loss = pynoisy.inverse.compute_pixel_loss(subspace, grf, damp=args.damp)
            losses.append(loss.expand_dims({'seed': [seed]}))
        losses = xr.concat(losses, dim='seed') if args.num_samples > 1 else losses[0]

        # Save datasets to NetCDF
        losses = script_utils.expand_dataset_dims(losses, config, params)
        losses.attrs.update(attrs)
        losses.to_netcdf(os.path.join(dirpath, 'loss{:04d}.nc'.format(i)))

    # Consolidate residuals into a single file (and delete temporary residual files)
    with xr.open_mfdataset(os.path.join(dirpath, 'loss*.nc')) as dataset:
        dataset = dataset.load().squeeze()
        dataset.attrs.update(attrs)
        dataset.to_netcdf(os.path.join(dirpath, '{}.benchmark.data.nc'.format(datetime)))

    for p in Path(dirpath).glob('loss*.nc'):
        p.unlink()

    # Copy script and config for reproducibility
    shutil.copy(__file__, os.path.join(dirpath, '{}.benchmark.script.py'.format(datetime)))
    shutil.copy(args.config, os.path.join(dirpath, '{}.benchmark.config.yaml'.format(datetime)))