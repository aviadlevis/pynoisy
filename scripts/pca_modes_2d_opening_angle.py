import xarray as xr
import pynoisy
import numpy as np
import argparse
from tqdm import tqdm
import time, os, json

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default='./opening_angles_modes',
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
                         default=8,
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
                         help='(default value: %(default)s) Number of data-points for the advection opening angle.')
    parser.add_argument('--degree',
                         type=int,
                         default=40,
                         help='(default value: %(default)s) Krylov dimensionality degree.')
    parser.add_argument('--blocksize',
                         type=int,
                         default=40,
                         help='(default value: %(default)s) Blocksize for LOBPCG.')
    parser.add_argument('--lobpcg_iter',
                        default=100,
                        help='(default value: %(default)s) maximum number of LOBPCG iterations.')
    parser.add_argument('--tol',
                        default=1.0,
                        help='(default value: %(default)s) Stop criteria for LOBPCG iterations.')
    parser.add_argument('--precond',
                        default=False,
                        action='store_true',
                        help='(default value: %(default)s) Use SMG precodintioning.')
    parser.add_argument('--std_scaling',
                        default=False,
                        action='store_true',
                        help='(default value: %(default)s) Scale std of resulting modes (and renormalize).')
    parser.add_argument('--deflate',
                        default=False,
                        action='store_true',
                        help='(default value: %(default)s) With or without deflation.')
    parser.add_argument('--verbose',
                        default=0,
                        help='(default value: %(default)s) Level of verbosity .')



    args = parser.parse_args()
    return args

# Parse input arguments
args = parse_arguments()

# Save a NetCDF file with groups
if os.path.isdir(args.output_dir) is False:
    os.mkdir(args.output_dir)

with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

filename = 'modes.LOBPCGiter{iter}.blocksize{blocksize}.degree{degree}.tol{tol}.scaling{scaling}.deflate{deflate}.{nt}x{nx}x{ny}'.format(
    iter=args.lobpcg_iter,
    blocksize=args.blocksize ,
    degree=args.degree,
    tol=args.tol,
    scaling=args.std_scaling,
    deflate=args.deflate,
    nt=args.nt,
    nx=args.nx,
    ny=args.ny
)
if args.deflate is False:
    filename = filename.replace('.degree{}.'.format(args.degree), '.')

output = os.path.join(args.output_dir, filename + '.{:03}.nc')

temporal_grid = np.linspace(-np.pi, np.pi, args.temporal_res)
spatial_grid = np.linspace(-np.pi/2, np.pi/2, args.spatial_res)

print('Starting iterations: deflation={}, blocksize={}, std_scaling={}, preconditioning={}'.format(
    args.deflate, args.blocksize, args.std_scaling, args.precond))
for i, spatial_angle in enumerate(tqdm(spatial_grid, desc='spatial_grid')):
    eigenvectors = []
    for j, temporal_angle in enumerate(tqdm(temporal_grid, desc='temporal_grid', leave=False)):
        diffusion = pynoisy.diffusion.general_xy(args.nx, args.ny, opening_angle=spatial_angle)
        advection = pynoisy.advection.general_xy(args.nx, args.ny, opening_angle=temporal_angle)
        solver = pynoisy.forward.HGRFSolver(args.nx, args.ny, advection, diffusion, seed=args.seed)

        if args.deflate:
            if (args.nx == 32) and (args.ny == 32) and (i >= 7) and (i <= 11) and (((j >= 4) and (j <= 6)) or ((j >= 13) and (j <= 15))):
                continue
            eigenvector = solver.get_eigenvectors_deflation(num_frames=args.nt, blocksize=args.blocksize , tol=args.tol,
                                                            degree=args.degree, precond=args.precond, verbose=args.verbose,
                                                            maxiter=args.lobpcg_iter, std_scaling=args.std_scaling,
                                                            n_jobs=args.n_jobs)

        else:
            eigenvector = solver.get_eigenvectors(num_frames=args.nt, blocksize=args.blocksize , tol=args.tol,
                                                  precond=args.precond, verbose=args.verbose, n_jobs=args.n_jobs,
                                                  maxiter=args.lobpcg_iter, std_scaling=args.std_scaling)

        eigenvector = eigenvector.expand_dims({'temporal_angle': [temporal_angle]})
        eigenvectors.append(eigenvector)

    eigenvectors = xr.concat(eigenvectors, dim='temporal_angle').expand_dims(spatial_angle=[spatial_angle])
    eigenvectors.attrs = {
        'runname': 'modes for spatial and temporal opening angles',
        'file_num': '{:03} / {:03}'.format(i, len(spatial_grid)-1),
        'date': time.strftime("%d-%b-%Y-%H:%M:%S"),
        'lobpcg_iter': args.lobpcg_iter,
        'blocksize': args.blocksize ,
        'preconditioning': 'True' if args.precond else 'False',
        'deflation': 'True' if args.deflate else 'False',
        'std_scaling':  'True' if args.std_scaling else 'False',
        'initial_seed': args.seed,
        'tol': args.tol,
        'n_jobs': args.n_jobs
    }
    eigenvectors.to_netcdf(output.format(i), mode='w')
