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
                        default=np.inf,
                        help='(default value: %(default)s) Modes degree.')
    parser.add_argument('--num_grid',
                         type=int,
                         default=5,
                         help='(default value: %(default)s) num_grid**2 number of true opening angle points.')

    args = parser.parse_args()
    return args



# Parse input arguments
args = parse_arguments()

directory = args.directory
startswith = args.startswith

files = [file for file in glob.glob(os.path.join(directory, '*.nc')) if file.split('/')[-1].startswith(startswith)]

if (startswith == 'vismodes'):
    from pynoisy import eht_functions as ehtf
    modes = pynoisy.utils.read_complex(files[0])
    obs = ehtf.load_obs(modes.array_path, modes.uvfits_path)
    nt, nx, ny, fov = modes.modes_nt, modes.modes_nx, modes.modes_ny, modes.fov
elif (startswith == 'modes'):
    modes = xr.load_dataset(files[0])
    nt, nx, ny =  modes.t.size, modes.x.size, modes.y.size

seed = args.seed
num_grid = args.num_grid

true_spatial_angle = np.linspace(-1.2, 1.2, num_grid)
true_temporal_angle = np.linspace(-3.14, 3.14, num_grid, endpoint=False)

residual_stats = []
for spatial_angle in tqdm(true_spatial_angle, desc='true spatial angle'):
    residuals = []
    for temporal_angle in tqdm(true_temporal_angle, desc='true temporal angle'):
        solver = pynoisy.forward.HGRFSolver.flat_variability(
            nx, ny, temporal_angle, spatial_angle, seed=args.seed
        )
        measurements = solver.run(num_frames=nt, n_jobs=args.n_jobs, verbose=False)

        if (startswith=='vismodes'):
            movie = ehtf.xarray_to_hdf5(measurements, obs, fov=fov, flipy=False)
            meas_obs = movie.observe_same_nonoise(obs)
            measurements = meas_obs.data['vis']

        output = pynoisy.utils.compute_residual(files, measurements, args.degree)
        residuals.append(output.expand_dims({'true_temporal_angle': [temporal_angle],
                                             'true_spatial_angle': [spatial_angle]}))
    residual_stats.append(xr.concat(residuals, dim='true_temporal_angle'))
residual_stats = xr.concat(residual_stats, dim='true_spatial_angle')

# update attributes
residual_stats.attrs = modes.attrs
residual_stats.attrs.update(
    file_num=len(files),
    directory=directory,
    measurement_seed=seed
)

# Save output NetCDF
residual_stats.to_netcdf(
    os.path.join(directory, 'residuals.{}.stats.num_spatial{}.num_temporal{}.seed{}.degree{}.nc'.format(
        startswith, residual_stats.true_spatial_angle.size, residual_stats.true_temporal_angle.size,
        seed, int(residual_stats.deg))))
