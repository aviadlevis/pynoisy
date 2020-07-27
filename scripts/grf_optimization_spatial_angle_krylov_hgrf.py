import os, time, json
import numpy as np
import argparse
import pynoisy
import xarray as xr
from pynoisy.inverse import SummaryWriter, Optimizer, ForwardOperator, ObjectiveFunction,  CallbackFn

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        help='(default value: %(default)s) Path to load pre-computed solver and measurements.')
    parser.add_argument('--args_path',
                        help='(default value: %(default)s) Path to load arguments from json.')
    parser.add_argument('--logdir',
                        default='runs/hgrf',
                        help='(default value: %(default)s) Path to directory and file name.')
    parser.add_argument('--run_name',
                        help='(default value: %(default)s) Name of the run.')
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
                         help='(default value: %(default)s) Krylov dimensionality reduction degree.')
    parser.add_argument('--solver_type',
                         type=str,
                         choices=['PCG', 'SMG'],
                         default='PCG',
                         help='(default value: %(default)s) HGRF solver typer.')
    parser.add_argument('--maxiter',
                         type=int,
                         default=50,
                         help='(default value: %(default)s) Maximum solver iterations.')
    parser.add_argument('--seed',
                         type=int,
                         default=None,
                         help='(default value: %(default)s) Seed for random number generator. None gives a random seed.')
    args = parser.parse_args()
    return args

# Parse input arguments and overwrite from file if path specified
args = parse_arguments()
if args.args_path is not None:
    with open(args.args_path, 'r') as file:
        args.__dict__ = json.load(file)

# Generate or Load synthetic measurements
if args.load_path is None:
    advection_true = pynoisy.advection.general_xy(args.nx, args.ny)
    diffusion_true = pynoisy.diffusion.general_xy(args.nx, args.ny)
    solver = pynoisy.forward.HGRFSolver(args.nx, args.ny, advection_true, diffusion_true, solver_type=args.solver_type, seed=args.seed)
    measurements = solver.run(maxiter=args.maxiter, num_frames=args.nt)
else:
    measurements = xr.load_dataarray(os.path.join(args.load_path,'measurements.nc'))
    solver = pynoisy.forward.HGRFSolver.from_netcdf(os.path.join(args.load_path, 'ground_truth.nc'))

# Initialize mask of inverse solver
run_name =  'spatial_angle_init[zero]_seed[{seed}]'.format(seed=solver.seed) if args.run_name is None else args.run_name
logdir = os.path.join(args.logdir, run_name)
writer = SummaryWriter(logdir=logdir)

solver.save(path=os.path.join(logdir, 'ground_truth.nc'))
solver.reseed()
solver.params['mask'] = solver.params.r < 0.5 - 2.0 / solver.params.dims['x']
solver.params.attrs['num_unknowns'] = solver.params.mask.sum().data
initial_state = np.zeros(shape=solver.params.num_unknowns)

adjoint_fn = forward_fn = lambda source: solver.run(source=source, maxiter=args.maxiter, verbose=False)
gradient_fn = pynoisy.inverse.spatial_angle_gradient(solver)
get_state_fn = pynoisy.inverse.spatial_angle_get_state(solver)
set_state_fn = pynoisy.inverse.spatial_angle_set_state(solver)

forward_op = ForwardOperator.krylov(
    forward_fn=forward_fn,
    adjoint_fn=adjoint_fn,
    gradient_fn=gradient_fn,
    set_state_fn=set_state_fn,
    get_state_fn=get_state_fn,
    measurements=measurements,
    degree=args.deg
)
objective_fn = ObjectiveFunction.l2(measurements, forward_op)

# Define callback functions for routine checks/analysis of the state
writer.average_image('average_frame/measurements', measurements)
writer.spatial_angle('diffusion/true_angle', solver.diffusion.spatial_angle, solver.params.mask)
callback_fn = [
    CallbackFn(lambda: writer.add_scalar('Loss', objective_fn.loss, optimizer.iteration)),
    CallbackFn(lambda: writer.spatial_angle('diffusion/estimate_angle', solver.diffusion.spatial_angle, solver.params.mask, optimizer.iteration)),
    CallbackFn(lambda: writer.average_image('average_frame/estimate', solver.run(maxiter=args.maxiter, seed=measurements.seed, verbose=False), optimizer.iteration), ckpt_period=5*60),
    CallbackFn(lambda: optimizer.save_checkpoint(solver, logdir), ckpt_period=1*60*60)
]
options={'maxiter': 1000, 'maxls': 100, 'disp': True, 'gtol': 1e-16, 'ftol': 1e-16}
optimizer = Optimizer(objective_fn, callback_fn=callback_fn, options=options)

# Export arguments to json and save measurements and intial state
with open(os.path.join(logdir, 'args.txt'), 'w') as file:
    json.dump(args.__dict__, file, indent=2)
measurements.to_netcdf(os.path.join(logdir, 'measurements.nc'))
optimizer.save_checkpoint(solver, logdir, name='initial_state.nc')


result = optimizer.minimize(initial_state=initial_state)
optimizer.save_checkpoint(solver, logdir, name='final_result.nc')
writer.close()


