import xarray as xr
import pynoisy
import numpy as np
import argparse
from tqdm import tqdm
import os, glob

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory',
                        default='opening_angle_modes',
                        help='(default value: %(default)s) Path to input / output directory.')
    parser.add_argument('--startswith',
                        default='modes',
                        help='(default value: %(default)s) Modes file names start with this string.')
    parser.add_argument('--n_jobs',
                         type=int,
                         default=4,
                         help='(default value: %(default)s) Number of jobs.')
    parser.add_argument('--seed',
                         type=int,
                         default=27669,
                         help='(default value: %(default)s) measurements seed.')
    parser.add_argument('--degree',
                         type=int,
                         help='measurements seed.')
    parser.add_argument('--num_grid',
                         type=int,
                         default=5,
                         help='(default value: %(default)s) num_grid**2 number of true opening angle points.')

    args = parser.parse_args()
    return args

def flat_variability_solver(nx, ny, temporal_angle, spatial_angle, seed):
    advection = pynoisy.advection.general_xy(nx, ny, opening_angle=temporal_angle)
    diffusion = pynoisy.diffusion.general_xy(nx, ny, opening_angle=spatial_angle)
    diffusion.correlation_time[:] = diffusion.correlation_time.mean()
    diffusion.correlation_length[:] = diffusion.correlation_length.mean()
    advection.magnitude[:] = 0.2
    advection.noisy_methods.update_vx_vy()
    solver = pynoisy.forward.HGRFSolver(nx, ny, advection, diffusion, seed=seed)
    return solver

def compute_residual(files, measurements, degree):
    output = []
    for file in tqdm(files, leave=False):
        residual = []
        modes = xr.load_dataset(file)
        for temporal_angle in modes.temporal_angle:
            eigenvectors = modes.sel(temporal_angle=temporal_angle).eigenvectors
            projection_degree = min(degree, eigenvectors.deg.size)

            projection_matrix = eigenvectors.isel(
                deg=slice(projection_degree)).noisy_methods.get_projection_matrix()
            projection = pynoisy.utils.least_squares_projection(measurements, projection_matrix)
            res = np.linalg.norm(measurements - projection) ** 2
            res = xr.DataArray(
                [res], dims='temporal_angle',
                coords={'temporal_angle': [temporal_angle]}).expand_dims(
                deg=[degree], spatial_angle=eigenvectors.spatial_angle)
            residual.append(res)
        output.append(xr.concat(residual, dim='temporal_angle').sortby('temporal_angle'))
    output = xr.concat(output, dim='spatial_angle').sortby('spatial_angle')
    return output

# Parse input arguments
args = parse_arguments()

directory = args.directory
startswith = args.startswith
files = [file for file in glob.glob(os.path.join(directory, '*.nc')) if file.split('/')[-1].startswith(startswith)]
modes = xr.load_dataset(files[0])
nt, nx, ny =  modes.t.size, modes.x.size, modes.y.size

seed = args.seed
num_grid = args.num_grid
enum_grid = args.num_grid

true_spatial_angle = np.linspace(-1.5, 1.5, num_grid)
true_temporal_angle = np.linspace(-3.1, 3.1, num_grid)

residual_stats = []
for spatial_angle in tqdm(true_spatial_angle, desc='true spatial angle'):
    residuals = []
    for temporal_angle in tqdm(true_temporal_angle, desc='true temporal angle'):
        solver = flat_variability_solver(nx, ny, temporal_angle, spatial_angle, seed)
        measurements = solver.run(num_frames=nt, n_jobs=args.n_jobs, verbose=False)
        output = compute_residual(files, measurements, args.degree)
        residuals.append(output.expand_dims({'true_temporal_angle': [temporal_angle],
                                             'true_spatial_angle': [spatial_angle]}))
    residual_stats.append(xr.concat(residuals, dim='true_temporal_angle'))
residual_stats = xr.concat(residual_stats, dim='true_spatial_angle')

# update attributes
residual_stats.attrs = modes.attrs
residual_stats.attrs.update(
    file_num=len(files),
    directory=directory,
    measurement_seed=measurements.seed
)

# Save output NetCDF
residual_stats.to_netcdf(
    os.path.join(directory, 'residuals.stats.num_spatial{}.num_temporal{}.seed{}.degree{}.nc'.format(
        residual_stats.true_spatial_angle.size, residual_stats.true_temporal_angle.size, seed, args.degree)))
