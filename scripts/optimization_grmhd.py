import os
import numpy as np
import pynoisy
import xarray as xr
from pynoisy.inverse import *
from pynoisy import utils
from joblib import Parallel, delayed
import scipy as sci
from scipy.sparse.linalg import lsqr


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

def set_disk_mask(solver):
    solver.params['mask'] = solver.params.r < 0.5 - 2.0 / solver.params.dims['x']
    solver.params.attrs['num_unknowns'] = solver.params.mask.sum().data
    return solver

def plot_envelope(envelope):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    envelope.plot(ax=ax[0])
    envelope.sel(x=0).plot(ax=ax[1])
    plt.tight_layout()

def estimate_envelope(grf, measurements, amplitude=1.0):
    num_frames = measurements.sizes['t']
    image_shape = (measurements.sizes['x'], measurements.sizes['y'])
    b = measurements.data.ravel()
    diags = np.exp(-amplitude * grf).data.reshape(num_frames, -1)
    A = sci.sparse.diags(diags, offsets=np.arange(num_frames) * -diags.shape[1],
                         shape=[diags.shape[1] * diags.shape[0], diags.shape[1]])
    sol = lsqr(A, b)[0]
    envelope = pynoisy.envelope.grid(data=sol.reshape(image_shape))
    return envelope

run_name = 'zero_init'

# Load measurements
load_path = 'runs/GRMHD/02/'
measurements = xr.load_dataarray(os.path.join(load_path,'measurements.nc'))

# Load measurements
advection = pynoisy.advection.disk(direction='ccw')
diffusion = pynoisy.diffusion.ring(opening_angle=-0.8)
diffusion.principle_angle[:] = 0.0
solver = pynoisy.forward.NoisySolver(advection, diffusion)
solver = set_disk_mask(solver)

forward_fn = lambda source: solver.run_symmetric(source, verbose=False)
adjoint_fn = lambda source: solver.run_adjoint(source, verbose=False)
gradient_fn = lambda forward, adjoint: compute_gradient(solver, forward, adjoint)
get_state_fn = lambda: solver.diffusion.principle_angle.values[solver.params.mask]
set_state_fn = lambda state: set_state(solver, state)

amplitude = 1.0
krylov_degree = 8
max_admm_iter = 30

logdir = os.path.join(load_path, run_name)
writer = SummaryWriter(logdir=logdir)
writer.average_image('average_frame/measurements', measurements)
callback_fn = [
    CallbackFn(lambda: writer.add_scalar('Loss', objective_fn.loss, optimizer.iteration)),
    CallbackFn(lambda: writer.diffusion('diffusion/estimate', solver.diffusion, solver.params.mask, optimizer.iteration, envelope=envelope)),
    CallbackFn(lambda: writer.principle_angle('diffusion/estimate_angle_no_alpha', solver.diffusion.principle_angle, solver.params.mask, optimizer.iteration)),
    CallbackFn(lambda: optimizer.save_checkpoint(solver, logdir), ckpt_period=1*60*60),
]
forward_op = ForwardOperator.krylov(
    forward_fn, adjoint_fn, gradient_fn, set_state_fn, get_state_fn, measurements, krylov_degree
)

num_iterations = 0
state = forward_op.get_state()
solver.save(os.path.join(logdir, 'initial_state.nc'))
options = {'maxiter': 100, 'maxls': 30, 'disp': True, 'gtol': 1e-16, 'ftol': 1e-16}
for i in range(max_admm_iter):
    prev_state = state
    envelope = estimate_envelope(forward_op(state), measurements, amplitude)
    writer.envelope('envelope/estimate', envelope, global_step=i)

    measurements_grf = np.log(
        envelope.where(envelope > 0) /
        measurements.where(measurements > 0)).transpose('t', 'x', 'y', transpose_coords=False).fillna(0.0)

    forward_op_grf = ForwardOperator.krylov(
        forward_fn, adjoint_fn, gradient_fn, set_state_fn, get_state_fn, measurements_grf, krylov_degree
    )
    objective_fn = ObjectiveFunction.l2(measurements_grf, forward_op_grf)
    optimizer = Optimizer(objective_fn, callback_fn=callback_fn, options=options)
    state = optimizer.minimize(initial_state=state, iteration_step=num_iterations).x
    num_iterations = optimizer.iteration

    # Termination criteria
    if np.allclose(state, prev_state, rtol=1e-2, atol=1e-3):
        break
    else:
        optimizer.save_checkpoint(solver, logdir, name='admm_iter{}_solver.nc'.format(i))
        envelope.to_netcdf(os.path.join(logdir, 'admm_iter{}_envelope.nc'.format(i)))

optimizer.save_checkpoint(solver, logdir, name='final_solver.nc')
envelope.to_netcdf(os.path.join(logdir, 'final_envelope.nc'))
writer.close()