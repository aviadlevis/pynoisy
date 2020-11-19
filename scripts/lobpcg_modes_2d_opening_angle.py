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
    parser.add_argument('--spatial_res',
                         type=int,
                         default=50,
                         help='(default value: %(default)s) Number of data-points for the diffusion opening angle.')
    parser.add_argument('--temporal_res',
                         type=int,
                         default=50,
                         help='(default value: %(default)s) Number of data-points for the advection opening angle.')
    parser.add_argument('--degree',
                         type=int,
                         default=24,
                         help='(default value: %(default)s) Krylov dimensionality degree.')
    parser.add_argument('--lobpcg_iter',
                        default=50,
                        help='(default value: %(default)s) maximum number of LOBPCG iterations.')
    parser.add_argument('--precond',
                        default=False,
                        help='(default value: %(default)s) Use SMG precodintioning.')
    args = parser.parse_args()
    return args

# Parse input arguments
args = parse_arguments()

# Save a NetCDF file with groups
if os.path.isdir(args.output_dir) is False:
    os.mkdir(args.output_dir)

with open(os.path.join(args.output_dir, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

filename = 'modes1.LOBPCGiter{iter}.degree{degree}.precond_{precond}.{nt}x{nx}x{ny}'.format(
    iter=args.lobpcg_iter,
    degree=args.degree,
    precond=args.precond,
    nt=args.nt,
    nx=args.nx,
    ny=args.ny
)
output = os.path.join(args.output_dir, filename + '.{:03}.nc')

advection = pynoisy.advection.general_xy(args.nx, args.ny)
diffusion = pynoisy.diffusion.general_xy(args.nx, args.ny)
solver = pynoisy.forward.HGRFSolver(args.nx, args.ny, advection, diffusion)
grf = solver.run(num_frames=args.nt, n_jobs=4, verbose=False)


#temporal_grid = np.linspace(-np.pi, np.pi, args.temporal_res)
#spatial_grid = np.linspace(-np.pi/2, np.pi/2, args.spatial_res)
temporal_grid = np.linspace(-np.pi + 2 * np.pi / 38, np.pi - 2 * np.pi / 38, 19)
spatial_grid = np.linspace(-np.pi/2 + np.pi / 38, np.pi/2 - np.pi / 38, 19)

for i, spatial_angle in enumerate(tqdm(spatial_grid, desc='spatial_grid')):
    eigenvectors = []
    for j, temporal_angle in enumerate(tqdm(temporal_grid, desc='temporal_grid', leave=False)):
        solver.update_diffusion(pynoisy.diffusion.general_xy(solver.nx, solver.ny, opening_angle=spatial_angle))
        solver.update_advection(pynoisy.advection.general_xy(solver.nx, solver.ny, opening_angle=temporal_angle))
        eigenvector = solver.get_eigenvectors(grf, degree=args.degree, precond=args.precond, verbose=False, maxiter=args.lobpcg_iter)
        eigenvector = eigenvector.eigenvectors.expand_dims({'temporal_angle': [temporal_angle]})
        eigenvectors.append(eigenvector)

    eigenvectors = xr.concat(eigenvectors, dim='temporal_angle')
    eigenvectors.attrs = {
        'runname': 'modes for spatial and temporal opening angles',
        'file_num': '{:03} / {:03}'.format(i, len(spatial_grid)-1),
        'date': time.strftime("%d-%b-%Y-%H:%M:%S"),
        'lobpcg_iter': args.lobpcg_iter,
        'preconditioning': 'True' if args.precond else 'False'
    }
    eigenvectors.to_netcdf(output.format(i), mode='w')
