import pynoisy
import numpy as np
import argparse
from tqdm import tqdm
import xarray as xr
import h5py
import os
import time
import scipy as sci
from scipy.sparse.linalg import lsqr


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

def load_grmhd(filepath, initial_frame, nt, nx, ny):
    filename =  os.path.abspath(filepath).split('/')[-1][:-3]
    assert initial_frame >= 0, 'Negative initial frame'
    with h5py.File(filepath, 'r') as file:
        frames = file['I'][:]
    nt_, nx_, ny_ = frames.shape
    assert (initial_frame + nt) < nt_, \
        'Final frame: {} is out of bounds for input movie of length: {}'.format(initial_frame + nt, nt_)

    grid = pynoisy.utils.get_grid(nx_, ny_)
    measurements = xr.DataArray(data=frames,
                                coords={'x': grid.x,  'y': grid.y, 't': np.linspace(0, 0.1, nt_)},
                                dims=['t', 'x', 'y'],
                                attrs={'GRMHD': filename})

    measurements = measurements[initial_frame:initial_frame + nt].interp_like(pynoisy.utils.get_grid(nx, ny))
    measurements.attrs.update({'initial_frame': initial_frame})
    return measurements

def estimate_envelope(grf, measurements, amplitude=1.0):
    num_frames = measurements.sizes['t']
    image_shape = (measurements.sizes['x'], measurements.sizes['y'])
    b = measurements.data.ravel()
    diagonals = np.exp(-amplitude * grf).data.reshape(num_frames, -1)
    A = sci.sparse.diags(diagonals, offsets=np.arange(num_frames) * -diagonals.shape[1],
                         shape=[diagonals.shape[1]*diagonals.shape[0], diagonals.shape[1]])
    sol = lsqr(A,b)[0]
    envelope = pynoisy.envelope.grid(data=sol.reshape(image_shape)).clip(min=0.0)
    return envelope


# Parse input arguments
args = parse_arguments()

# Load and resample measurements
measurements = load_grmhd(args.filepath, args.initial_frame, args.nt, args.nx, args.ny)

# Generate initial GRF for envelope estimation
# In general the envelope should be refined with updated GRF parameters, however, it is robust enough
# For a single estimation with some initial GRF as a proxy
advection = pynoisy.advection.general_xy(measurements.x.size, measurements.y.size, direction='cw')
diffusion = pynoisy.diffusion.general_xy(measurements.x.size, measurements.y.size, opening_angle=-1.2)
solver = pynoisy.forward.HGRFSolver(measurements.x.size, measurements.y.size, advection, diffusion)
grf = solver.run(num_frames=measurements.t.size, n_jobs=args.n_jobs)
envelope = estimate_envelope(grf, measurements)

# Define the GRF measurements as a logarithm, add random noise at pixel (0,0) for stability
measurements_grf = np.log((envelope.where(envelope>1e-8) / measurements.where(measurements>1e-8))).transpose(
    't', 'x', 'y', transpose_coords=False).fillna(0.0)
measurements_grf.loc[{'x': 0, 'y':0}] = np.random.randn(measurements_grf.t.size)


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
        'krylov_degree': args.deg
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
