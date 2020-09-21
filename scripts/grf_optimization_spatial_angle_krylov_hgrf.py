import os, json, time
import numpy as np
import argparse
import pynoisy
import xarray as xr
from pynoisy.inverse import SummaryWriter, Optimizer, ForwardOperator, ObjectiveFunction,  CallbackFn
import shutil

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver_path',
                        help='Path to load solver.')
    parser.add_argument('--meas_path',
                        help='(Path to load measurements.')
    parser.add_argument('--args_path',
                        help='Path to load arguments from json.')
    parser.add_argument('--logdir',
                        default='runs/hgrf',
                        help='(default value: %(default)s) Path to directory and file name.')
    parser.add_argument('--run_name',
                        help='Name of the run.')
    parser.add_argument('--nx',
                         type=int,
                         default=64,
                         help='(default value: %(default)s) Number of x grid points.')
    parser.add_argument('--ny',
                         type=int,
                         default=64,
                         help='(default value: %(default)s) Number of y grid points.')
    parser.add_argument('--nt',
                         type=int,
                         default=64,
                         help='(default value: %(default)s) Number of video frames.')
    parser.add_argument('--deg',
                         type=int,
                         default=8,
                         help='(default value: %(default)s) Krylov dimensionality degree.')
    parser.add_argument('--solver_type',
                         type=str,
                         choices=['PCG', 'SMG'],
                         default='SMG',
                         help='(default value: %(default)s) HYPRE solver type.')
    parser.add_argument('--maxiter',
                         type=int,
                         default=50,
                         help='(default value: %(default)s) Maximum solver iterations.')
    parser.add_argument('--n_jobs',
                         type=int,
                         default=4,
                         help='(default value: %(default)s) Number of parallel jobs.')
    parser.add_argument('--seed',
                         type=int,
                         default=None,
                         help='(default value: %(default)s) Seed for random number generator. None gives a random seed.')
    parser.add_argument('--optimizer',
                         default='GD',
                         help='(default value: %(default)s) Optimizer type (see Optimizer class for more details on the types).')
    parser.add_argument('--step_size',
                         type=float,
                         default=0.1,
                         help='(default value: %(default)s) Initial step size used for GD (Gradient Descent).')
    parser.add_argument('--dct_cutoff',
                         type=int,
                         default=32,
                         help='(default value: %(default)s) Cutoff index for the DCT transform.')
    args = parser.parse_args()
    return args

# Parse input arguments and overwrite from file if path specified
args = parse_arguments()
if args.args_path is not None:
    with open(args.args_path, 'r') as file:
        args.__dict__ = json.load(file)


# Generate or Load synthetic measurements
if args.meas_path is None:
    advection_true = pynoisy.advection.general_xy(args.nx, args.ny)
    diffusion_true = pynoisy.diffusion.general_xy(args.nx, args.ny)
    advection_true.vx[:] = 0.0
    advection_true.vy[:] = 0.0
    solver = pynoisy.forward.HGRFSolver(args.nx, args.ny, advection_true, diffusion_true, solver_type=args.solver_type, seed=args.seed)
    measurements = solver.run(maxiter=args.maxiter, num_frames=args.nt, n_jobs=args.n_jobs)
else:
    measurements = xr.load_dataarray(os.path.join(args.meas_path))

# Setup write for tensorboard logging
run_name =  'krylov_spatial_angle_init[zero]_seed[{seed}]_opt[{opt}]_step[{step}]_dctcutoff[{dct_cutoff}]'.format(
    seed=solver.seed, opt=args.optimizer, step=args.step_size, dct_cutoff=args.dct_cutoff) if args.run_name is None else args.run_name
logdir = os.path.join(args.logdir, run_name)
writer = SummaryWriter(logdir=logdir)
if args.meas_path is None:
    solver.save(path=os.path.join(logdir, 'ground_truth.nc'))
    solver.reseed()

# Setup inverse solver:
# 1. mask for the reconstruction
# 2. initial state
if args.solver_path is None:
    mask = xr.ones_like(solver.diffusion.spatial_angle, dtype=np.bool)
    solver.params['mask'] = mask
    solver.params.attrs['num_unknowns'] = solver.params.mask.sum().data
    initial_state = np.zeros(shape=solver.params.num_unknowns)
else:
    solver = pynoisy.forward.HGRFSolver.from_netcdf(args.solver_path)
    initial_state = None

forward_fn = lambda source: solver.run(source=source, maxiter=args.maxiter, verbose=False, n_jobs=args.n_jobs, std_scaling=False)
adjoint_fn = lambda source: solver.run(source=source, maxiter=args.maxiter, verbose=False, n_jobs=args.n_jobs, std_scaling=True)
gradient_fn = lambda forward, adjoint: solver.get_spatial_angle_gradient(forward, adjoint).values[solver.params.mask==True]
get_state_fn = pynoisy.inverse.spatial_angle_get_state(solver)
set_state_fn = pynoisy.inverse.spatial_angle_set_state(solver)
k_matrix_fn = lambda source, degree: solver.run(source=source, num_frames=args.nt, n_jobs=args.n_jobs, verbose=False, nrecur=degree, std_scaling=False).data.reshape(degree, -1)

# DCT blurring to state
set_state = pynoisy.inverse.spatial_angle_set_state(solver)
import scipy.fftpack as fftpack
def dct_filtering(state, cutoff_idx):
    set_state(state)
    angle = np.exp(np.complex(0, 1) * solver.diffusion.spatial_angle)
    dct = fftpack.dctn(angle, norm='ortho')
    dct[cutoff_idx:] = 0.0
    dct[:,cutoff_idx:] = 0.0
    solver.diffusion.spatial_angle.values[solver.params.mask==True] = np.angle(fftpack.idctn(dct, norm='ortho'))[solver.params.mask==True]

set_state_dct_fn = lambda state: dct_filtering(state, cutoff_idx=args.dct_cutoff)

forward_op = ForwardOperator.krylov(
    forward_fn=forward_fn,
    adjoint_fn=adjoint_fn,
    gradient_fn=gradient_fn,
    set_state_fn=set_state_dct_fn,
    get_state_fn=get_state_fn,
    measurements=measurements,
    degree=args.deg,
    k_matrix_fn=k_matrix_fn
)
objective_fn = ObjectiveFunction.l2(measurements, forward_op)

# Define callback functions for routine checks/analysis of the state
writer.average_image('average_frame/measurements', measurements)
if args.meas_path is None:
    writer.spatial_angle('diffusion/true_angle', diffusion_true.spatial_angle)
callback_fn = [
    CallbackFn(lambda: writer.add_scalar('Loss', objective_fn.loss, optimizer.iteration)),
    CallbackFn(lambda: writer.spatial_angle('diffusion/estimate_angle', solver.diffusion.spatial_angle, global_step=optimizer.iteration)),
    CallbackFn(lambda: writer.average_image('average_frame/estimate', solver.run(num_frames=args.nt, maxiter=args.maxiter, seed=measurements.seed, verbose=False), global_step=optimizer.iteration)),
    CallbackFn(lambda: optimizer.save_checkpoint(solver, logdir), ckpt_period=1*60*60)
]

if args.optimizer == 'GD':
    options={'maxiter': 1000, 'disp': True, 'initial_step_size': args.step_size, 'maxls': 20}
elif args.optimizer == 'L-BFGS-B':
    options={'maxiter': 1000, 'disp': True, 'ftol': 1e-16, 'gtol': 1e-16}
elif args.optimizer == 'CG':
    options = {'disp': True}
optimizer = Optimizer(objective_fn, callback_fn=callback_fn, options=options, method=args.optimizer)

# Export arguments to json and save measurements and intial state
with open(os.path.join(logdir, 'args.txt'), 'w') as file:
    json.dump(args.__dict__, file, indent=2)
measurements.to_netcdf(os.path.join(logdir, 'measurements.nc'))

initial_state = forward_op.get_state() if initial_state is None else initial_state
forward_op.set_state(initial_state)
optimizer.save_checkpoint(solver, logdir, name='initial_state.nc')

result = optimizer.minimize(initial_state=initial_state)
optimizer.save_checkpoint(solver, logdir, name='final_result.nc')
writer.close()

# Copy the script for reproducibility of the experiment
shutil.copy(__file__, os.path.join(logdir, '[{}]script.py'.format(time.strftime("%d-%b-%Y-%H:%M:%S"))))