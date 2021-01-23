import xarray as xr
import pynoisy
import numpy as np
import argparse
from tqdm import tqdm
import time, os, json
import shutil
import scipy.linalg

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default='./opening_angles_modes_subspace_iteration/',
                        help='(default value: %(default)s) Path to output directory.')
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
    parser.add_argument('--n_jobs',
                         type=int,
                         default=4,
                         help='(default value: %(default)s) Number of jobs.')
    parser.add_argument('--spatial_res',
                         type=int,
                         default=20,
                         help='(default value: %(default)s) Number of data-points for the diffusion opening angle.')
    parser.add_argument('--temporal_res',
                         type=int,
                         default=20,
                         help='(default value: %(default)s) Number of data-points for the advection opening angle.')
    parser.add_argument('--seed',
                         type=int,
                         default=5,
                         help='(default value: %(default)s) Measurement seed.')
    parser.add_argument('--blocksize',
                         type=int,
                         default=40,
                         help='(default value: %(default)s) Blocksize for LOBPCG.')
    parser.add_argument('--num_iterations',
                        default=10,
                        type=int,
                        help='(default value: %(default)s) maximum number of LOBPCG iterations.')
    parser.add_argument('--advection_magnitude',
                        default=0.2,
                        type=float,
                        help='(default value: %(default)s) Flat advection magnitude.')
    args = parser.parse_args()
    return args

def multiple_sources(solver, sources, n_jobs=4):
    output = []
    for source in sources:
        grf = solver.run(source=source, n_jobs=n_jobs, verbose=False)
        output.append(grf)
    return output

def compute_subspace(matrix_fn, input_vectors):
    subspace = matrix_fn(input_vectors)
    q, r = scipy.linalg.qr(subspace, mode='economic', overwrite_a=True)
    q_xr = xr.DataArray(q.T.reshape(input_vectors.shape), coords=input_vectors.coords)
    return q_xr

def randomized_subspace_iteration(matrix_fn, input_vectors, maxiter=5, svd=False):
    subspace = input_vectors
    for i in tqdm(range(maxiter), desc='subspace iteration', leave=False):
        subspace = compute_subspace(matrix_fn, subspace)
    if svd == True:
        subspace = matrix_fn(subspace)
        u, s, v = scipy.linalg.svd(subspace, full_matrices=False)
        subspace = xr.DataArray(u.T.reshape(input_vectors.shape), coords=input_vectors.coords)
    return subspace.rename(sample='deg')

matrix_fn = lambda sources: np.array([grf.data.ravel() for grf in multiple_sources(solver, sources, n_jobs=4)]).T

# Parse input arguments
args = parse_arguments()

# Save a NetCDF file with groups
if os.path.isdir(args.output_dir) is False:
    os.mkdir(args.output_dir)

with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

filename = 'flatmodes.subspace.iter{iter}.blocksize{blocksize}.{nt}x{nx}x{ny}'.format(
    iter=args.num_iterations,
    blocksize=args.blocksize ,
    nt=args.nt,
    nx=args.nx,
    ny=args.ny
)

output = os.path.join(args.output_dir, filename + '.{:03}.nc')
temporal_grid = np.linspace(-np.pi, np.pi, args.temporal_res)
spatial_grid = np.linspace(-np.pi/2, np.pi/2, args.spatial_res)

solver = pynoisy.forward.HGRFSolver.flat_variability(
    args.nx, args.ny, advection_magnitude=args.advection_magnitude, seed=args.seed
)
random_vectors = solver.sample_source(num_frames=args.nt, num_samples=args.blocksize)
for i, spatial_angle in enumerate(tqdm(spatial_grid, desc='spatial_grid')):
    eigenvectors = []
    for j, temporal_angle in enumerate(tqdm(temporal_grid, desc='temporal_grid', leave=False)):
        # advection = pynoisy.advection.general_xy(args.nx, args.ny, opening_angle=temporal_angle)
        # diffusion = pynoisy.diffusion.general_xy(args.nx, args.ny, opening_angle=spatial_angle)
        # solver = pynoisy.forward.HGRFSolver(measurements.x.size, measurements.y.size, advection, diffusion)

        solver = pynoisy.forward.HGRFSolver.flat_variability(
            args.nx, args.ny, temporal_angle=temporal_angle, spatial_angle=spatial_angle,
            advection_magnitude=args.advection_magnitude, seed=args.seed
        )

        eigenvector = randomized_subspace_iteration(matrix_fn, random_vectors, maxiter=args.num_iterations, svd=True)
        eigenvector = eigenvector.expand_dims({'temporal_angle': [temporal_angle]})
        eigenvectors.append(eigenvector)


    eigenvectors = xr.concat(eigenvectors, dim='temporal_angle').expand_dims(spatial_angle=[spatial_angle])
    eigenvectors.attrs = {
        'runname': 'flat correlation modes for spatial and temporal opening angles [flat variability]',
        'file_num': '{:03} / {:03}'.format(i, len(spatial_grid)-1),
        'date': time.strftime("%d-%b-%Y-%H:%M:%S"),
        'subspace_iter': args.num_iterations,
        'blocksize': args.blocksize ,
        'initial_seed': args.seed,
        'n_jobs': args.n_jobs,
        'advection_magnitude': args.advection_magnitude
    }
    eigenvectors.to_netcdf(output.format(i), mode='w')


filename = 'modes.subspace.iter{iter}.blocksize{blocksize}.{nt}x{nx}x{ny}'.format(
    iter=args.num_iterations,
    blocksize=args.blocksize ,
    nt=args.nt,
    nx=args.nx,
    ny=args.ny
)

output = os.path.join(args.output_dir, filename + '.{:03}.nc')
for i, spatial_angle in enumerate(tqdm(spatial_grid, desc='spatial_grid')):
    eigenvectors = []
    for j, temporal_angle in enumerate(tqdm(temporal_grid, desc='temporal_grid', leave=False)):
        advection = pynoisy.advection.general_xy(args.nx, args.ny, opening_angle=temporal_angle)
        diffusion = pynoisy.diffusion.general_xy(args.nx, args.ny, opening_angle=spatial_angle)
        solver = pynoisy.forward.HGRFSolver(args.nx, args.ny, advection, diffusion)
        eigenvector = randomized_subspace_iteration(matrix_fn, random_vectors, maxiter=args.num_iterations, svd=True)
        eigenvector = eigenvector.expand_dims({'temporal_angle': [temporal_angle]})
        eigenvectors.append(eigenvector)

    eigenvectors = xr.concat(eigenvectors, dim='temporal_angle').expand_dims(spatial_angle=[spatial_angle])
    eigenvectors.attrs = {
        'runname': 'modes for spatial and temporal opening angles [radial variability]',
        'file_num': '{:03} / {:03}'.format(i, len(spatial_grid)-1),
        'date': time.strftime("%d-%b-%Y-%H:%M:%S"),
        'subspace_iter': args.num_iterations,
        'blocksize': args.blocksize ,
        'initial_seed': args.seed,
        'n_jobs': args.n_jobs,
    }
    eigenvectors.to_netcdf(output.format(i), mode='w')

# Copy the script for reproducibility of the experiment
shutil.copy(__file__, os.path.join(args.output_dir, '[{}]script.py'.format(time.strftime("%d-%b-%Y-%H:%M:%S"))))