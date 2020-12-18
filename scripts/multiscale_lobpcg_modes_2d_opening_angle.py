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
    parser.add_argument('--lobpcg_iter',
                        default=200,
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

filename = 'multiscale.modes.LOBPCGiter{iter}.tol{tol}.scaling{scaling}'.format(
    iter=args.lobpcg_iter,
    tol=args.tol,
    scaling=args.std_scaling
)

output = os.path.join(args.output_dir, filename + '.{:03}.nc')

temporal_grid = np.linspace(-np.pi, np.pi, args.temporal_res)
spatial_grid = np.linspace(-np.pi/2, np.pi/2, args.spatial_res)

multi_res = [2, 4, 8, 16, 32, 64]
jobs = [1, 1, 1, 4, 8, 16]
blocksizes = []

for i, spatial_angle in enumerate(tqdm(spatial_grid, desc='spatial_grid')):
    eigenvectors = []
    for j, temporal_angle in enumerate(tqdm(temporal_grid, desc='temporal_grid', leave=False)):
        multi_res_modes = []
        for n_jobs, resolution in zip(jobs, multi_res):
            nt, nx, ny = resolution, resolution, resolution
            advection = pynoisy.advection.general_xy(nx, ny, opening_angle=temporal_angle)
            diffusion = pynoisy.diffusion.general_xy(nx, ny, opening_angle=spatial_angle)
            solver = pynoisy.forward.HGRFSolver(nx, ny, advection, diffusion, seed=args.seed)

            blocksize = int(2 * np.log2(nx * ny * nt))
            blocksizes.append(blocksize)
            modes = solver.get_eigenvectors(num_frames=nt, blocksize=blocksize , tol=args.tol,
                                              precond=args.precond, verbose=args.verbose, n_jobs=n_jobs,
                                              maxiter=args.lobpcg_iter, std_scaling=args.std_scaling)
            multi_res_modes.append(modes.expand_dims({'scale': [resolution]}))

        multi_scale_modes = []
        for modes in multi_res_modes:
            method = 'linear' if modes.scale < 4 else 'cubic'
            interp_modes = modes.interp(
                {'t': multi_res_modes[-1].coords['t'],
                 'x': multi_res_modes[-1].coords['x'],
                 'y': multi_res_modes[-1].coords['y']}, method=method)
            interp_modes.coords.update({'deg': range(interp_modes.deg.size)})
            multi_scale_modes.append(interp_modes)
        multi_scale_modes = xr.concat(multi_scale_modes, dim='scale').stack(msdeg=('scale', 'deg')).dropna('msdeg')

        multi_scale_modes = multi_scale_modes.expand_dims({'temporal_angle': [temporal_angle]})
        eigenvectors.append(multi_scale_modes)

    eigenvectors = xr.concat(eigenvectors, dim='temporal_angle').expand_dims(spatial_angle=[spatial_angle])
    eigenvectors.attrs = {
        'runname': 'multiscale modes for spatial and temporal opening angles',
        'file_num': '{:03} / {:03}'.format(i, len(spatial_grid)-1),
        'date': time.strftime("%d-%b-%Y-%H:%M:%S"),
        'lobpcg_iter': args.lobpcg_iter,
        'preconditioning': 'True' if args.precond else 'False',
        'std_scaling':  'True' if args.std_scaling else 'False',
        'initial_seed': args.seed,
        'tol': args.tol,
        'blocksizes_per_scale': blocksizes,
        'n_jobs_per_scale': jobs
    }
    eigenvectors.reset_index('msdeg').to_netcdf(output.format(i), mode='w')
