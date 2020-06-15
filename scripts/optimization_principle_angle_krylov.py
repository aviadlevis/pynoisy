import os
import numpy as np
import argparse
import pynoisy
import xarray as xr
from pynoisy.inverse import SummaryWriter, Optimizer, ForwardOperator, ObjectiveFunction, PriorFunction, CallbackFn

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path',
                        help='(default value: %(default)s) Path to load pre-computed solver and measurements.')
    parser.add_argument('--logdir',
                        default='runs',
                        help='(default value: %(default)s) Path to directory and file name.')
    parser.add_argument('--deg',
                         type=int,
                         default=8,
                         help='(default value: %(default)s) Krylov dimensionality reduction degree.')
    parser.add_argument('--seed',
                         type=int,
                         default=None,
                         help='(default value: %(default)s) Seed for random number generator. None gives a random seed.')
    parser.add_argument('--diffusion_model',
                         type=str,
                         choices=['ring', 'multivariate_gaussian'],
                         default='ring',
                         help='(default value: %(default)s) Diffusion model. If multivariate_gaussian, '
                              'correlation length (args.length_scale) is taken into account')
    parser.add_argument('--length_scale',
                         type=float,
                         default=0.1,
                         help='(default value: %(default)s) Correlation length for multivariate_gaussian diffusion model')
    parser.add_argument('--prior_weight',
                         type=float,
                         default=0.0,
                         help='(default value: %(default)s) Weight on the prior term')
    args = parser.parse_args()
    return args

def compute_gradient(solver, forward, adjoint, dx=1e-2):
    principle_angle = solver.diffusion.principle_angle.copy()
    source = solver.get_laplacian(forward)
    gradient = np.zeros(shape=solver.params.num_unknowns)
    for n, (i, j) in enumerate(zip(*np.where(solver.params.mask))):
        solver.diffusion.principle_angle[i, j] = principle_angle[i, j] + dx
        source_ij = solver.get_laplacian(forward) - source
        solver.diffusion.principle_angle[i, j] = principle_angle[i, j]
        source_ij = source_ij / dx
        gradient[n] += (adjoint * source_ij).mean()
    return gradient

def set_state(solver, state):
    solver.diffusion.principle_angle.values[solver.params.mask] = state

# Parse input arguments
args = parse_arguments()

if args.load_path is None:
    # Generate synthetic measurements
    advection_true = pynoisy.advection.disk()
    if args.diffusion_model == 'multivariate_gaussian':
        diffusion_true = pynoisy.diffusion.multivariate_gaussian(length_scale=args.length_scale)
    elif args.diffusion_model == 'ring':
        diffusion_true = pynoisy.diffusion.ring()
    solver = pynoisy.forward.NoisySolver(advection_true, diffusion_true, seed=args.seed)
    measurements = solver.run_symmetric()
else:
    # Load measurements and solver
    measurements = xr.load_dataarray(os.path.join(args.load_path,'measurements.nc'))
    solver = pynoisy.forward.NoisySolver.from_netcdf(os.path.join(args.load_path, 'ground_truth.nc'))

# Initialize mask of inverse solver
run_name =  'priniple_angle_init[zero]_krylovdeg[{deg}]_mask[disk]_prior[{prior}]_seed[{seed}]'.format(
    deg=args.deg, seed=solver.seed, prior=args.prior_weight)
logdir = os.path.join(args.logdir, run_name)
writer = SummaryWriter(logdir=logdir)

solver.save(path=os.path.join(logdir, 'ground_truth.nc'))
solver.reseed()
solver.params['mask'] = solver.params.r < 0.5 - 2.0 / solver.params.dims['x']
solver.params.attrs['num_unknowns'] = solver.params.mask.sum().data
initial_state = np.zeros(shape=solver.params.num_unknowns)

forward_fn = lambda source: solver.run_symmetric(source, verbose=False)
adjoint_fn = lambda source: solver.run_adjoint(source, verbose=False)
gradient_fn = lambda forward, adjoint: compute_gradient(solver, forward, adjoint)
get_state_fn = lambda: solver.diffusion.principle_angle.values[solver.params.mask]
set_state_fn = lambda state: set_state(solver, state)

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

if args.prior_weight:
    if solver.diffusion.diffusion_model == 'multivariate_gaussian':
        flat_mask = np.array(solver.params.mask).ravel()
        covariance_mask = np.dot(flat_mask[:, None], flat_mask[None])
        cov = solver.diffusion.covariance.values[covariance_mask].reshape(
            solver.params.num_unknowns, solver.params.num_unknowns)
        prior_fn = PriorFunction.mahalanobis(mean=0.0, cov=cov, scale=args.prior_weight)
        loss_callback_fn = CallbackFn(lambda: writer.add_scalars(
            'Loss', {'total': objective_fn.loss + prior_fn.loss, 'datafit': objective_fn.loss, 'prior': prior_fn.loss}, optimizer.iteration))
    else:
        raise AttributeError('Prior function not implemented for diffusion_model={}'.format(args.diffusion_model))
else:
    prior_fn = None
    loss_callback_fn = CallbackFn(lambda: writer.add_scalar('Loss', objective_fn.loss, optimizer.iteration))


# Define callback functions for routine checks/analysis of the state
writer.average_image('average_frame/measurements', measurements)
writer.principle_angle('diffusion/true_angle', solver.diffusion.principle_angle, solver.params.mask)
callback_fn = [
    loss_callback_fn,
    CallbackFn(lambda: writer.principle_angle('diffusion/estimate_angle', solver.diffusion.principle_angle, solver.params.mask, optimizer.iteration)),
    CallbackFn(lambda: writer.average_image('average_frame/estimate', solver.run_symmetric(verbose=False), optimizer.iteration), ckpt_period=5*60),
    CallbackFn(lambda: optimizer.save_checkpoint(solver, logdir), ckpt_period=1*60*60)
]

options={'maxiter': 10000, 'maxls': 100, 'disp': True, 'gtol': 1e-16, 'ftol': 1e-16}
optimizer = Optimizer(objective_fn, prior_fn=prior_fn, callback_fn=callback_fn, options=options)

measurements.to_netcdf(os.path.join(logdir, 'measurements.nc'))
optimizer.save_checkpoint(solver, logdir, name='initial_state.nc')
result = optimizer.minimize(initial_state=initial_state)
optimizer.save_checkpoint(solver, logdir, name='final_result.nc')
writer.close()
