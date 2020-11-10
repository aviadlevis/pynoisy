import xarray as xr
import pynoisy
import numpy as np
import argparse
from tqdm import tqdm
import time


def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath',
                        help='Input .h5 file of GRMHD movie.')
    parser.add_argument('--output',
                        default=None,
                        help='(default value: %(default)s) Path to output file. If None is provided than a default naming convention is used and the file is output at the current directory.')
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
    parser.add_argument('--initial_frame',
                         type=int,
                         default=0,
                         help='(default value: %(default)s) Initial frame used for cropping the movie.')
    parser.add_argument('--spatial_res',
                         type=int,
                         default=50,
                         help='(default value: %(default)s) Number of data-points for the diffusion opening angle.')
    parser.add_argument('--temporal_res',
                         type=int,
                         default=50,
                         help='(default value: %(default)s) Number of data-points for the advection opening angle.')
    parser.add_argument('--deg',
                         type=int,
                         default=8,
                         help='(default value: %(default)s) Krylov dimensionality degree.')
    parser.add_argument('--n_jobs',
                         type=int,
                         default=4,
                         help='(default value: %(default)s) Number of parallel jobs.')
    parser.add_argument('--envelope_threshold',
                         type=int,
                         default=1e-8,
                         help='(default value: %(default)s) Threshold for the envelope.')

    args = parser.parse_args()
    return args

def objective_fun(diffusion_angle, advection_angle, solver, measurements, degree=8, n_jobs=4):
    solver.update_diffusion(pynoisy.diffusion.general_xy(solver.nx, solver.ny, opening_angle=diffusion_angle))
    solver.update_advection(pynoisy.advection.general_xy(solver.nx, solver.ny, opening_angle=advection_angle))
    error, forward = krylov_error_fn(solver, measurements, degree, n_jobs)
    loss = (error ** 2).mean()
    return np.array(loss)

def krylov_error_fn(solver, measurements, degree, n_jobs):
    krylov = solver.run(source=measurements, nrecur=degree, verbose=0, std_scaling=False, n_jobs=n_jobs)
    k_matrix = krylov.data.reshape(degree, -1)
    result = np.linalg.lstsq(k_matrix.T, np.array(measurements).ravel(), rcond=-1)
    coefs, residual = result[0], result[1]
    random_field = np.dot(coefs.T, k_matrix).reshape(*measurements.shape)
    error = random_field - measurements
    return error, random_field

def resample_movie(movie, initial_frame, nt, nx, ny):
    assert initial_frame >= 0, 'Negative initial frame'
    assert (initial_frame + nt) < movie.t.size, 'Final frame: {} out of bounds for input of length: {}'.format(
        initial_frame + nt, movie.t.size)
    movie = movie[initial_frame:initial_frame + nt].interp_like(pynoisy.utils.get_grid(nx, ny))
    movie.attrs.update({'initial_frame': initial_frame})
    return movie

def random_center(measurements):
    measurements = measurements.fillna(0.0)
    measurements.loc[{'x': 0, 'y':0}] = np.random.randn(measurements.t.size)
    return measurements

# Parse input arguments
args = parse_arguments()

# Load and resample measurements
measurements = pynoisy.utils.load_grmhd(args.filepath)
measurements = resample_movie(measurements, args.initial_frame, args.nt, args.nx, args.ny)

# Define the GRF measurements as a logarithm, add random noise at pixel (0,0) for stability
measurements_grf = np.log(measurements.where(measurements > args.envelope_threshold))
measurements_grf = measurements_grf - measurements_grf.mean('t')
measurements_grf = random_center(measurements_grf)


advection = pynoisy.advection.general_xy(measurements.x.size, measurements.y.size, direction='cw')
diffusion = pynoisy.diffusion.general_xy(measurements.x.size, measurements.y.size, opening_angle=-1.2)
solver = pynoisy.forward.HGRFSolver(measurements.x.size, measurements.y.size, advection, diffusion)

# Generate the 2D Krylov loss manifold
temporal_grid = np.linspace(-np.pi, np.pi, args.temporal_res)
spatial_grid = np.linspace(-np.pi/2, np.pi/2, args.spatial_res)

loss = np.empty((args.spatial_res, args.temporal_res))
for i, spatial_angle in enumerate(tqdm(spatial_grid, desc='spatial_grid')):
    for j, temporal_angle in enumerate(tqdm(temporal_grid, desc='temporal_grid', leave=False)):
        loss[i, j] = objective_fun(spatial_angle, temporal_angle, solver, measurements_grf, args.deg, args.n_jobs)

# Create a dataset and save results
dataset = xr.Dataset(
    data_vars={'loss': (['spatial_angle', 'temporal_angle'], loss)},
    coords={'spatial_angle': spatial_grid, 'temporal_angle':temporal_grid},
    attrs={
        'GRMHD': measurements.GRMHD,
        'nx': measurements.x.size,
        'ny': measurements.y.size,
        'nt': measurements.t.size,
        'initial_frame': measurements.initial_frame,
        'krylov_degree': args.deg,
        'envelope_threshold': args.envelope_threshold
    }
)

output = './spatio_temporal_loss_{grmhd}_{nt}x{nx}x{ny}_{time}.nc'.format(
    grmhd=dataset.GRMHD,
    nt=dataset.nt,
    nx=dataset.nx,
    ny=dataset.ny,
    time=time.strftime("%d-%b-%Y-%H:%M:%S")
) if args.output is None else args.output


print('Saving dataset to file: {}'.format(output))
dataset.to_netcdf(output)
