import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import core
import pynoisy
import time, os
from pynoisy.inverse import SummaryWriter, Optimizer, ForwardOperator, ObjectiveFunction, CallbackFn


# Generate synthetic measurements
advection_true = pynoisy.advection.disk()
diffusion_true = pynoisy.diffusion.ring()
solver = pynoisy.forward.NoisySolver(advection_true, diffusion_true, seed=1773)
measurements = solver.run_symmetric()

"""
No noise diffusion angle
"""
def compute_gradient(solver, forward, adjoint, dx=1e-2):
    spatial_angle = solver.diffusion.spatial_angle.copy()
    source = solver.get_laplacian(forward)
    gradient = np.zeros((64,64))
    for i in range(2,62):
        for j in range(2,62):
            solver.diffusion.spatial_angle[i, j] = spatial_angle[i, j] + dx
            source_ij = solver.get_laplacian(forward) - source
            solver.diffusion.spatial_angle[i, j] = spatial_angle[i, j]
            source_ij = source_ij / dx
            gradient[i,j] += (adjoint * source_ij).mean()
    return gradient


forward_fn = lambda: solver.run_symmetric(verbose=False)
adjoint_fn = lambda source: solver.run_adjoint(source, verbose=False)
gradient_fn = lambda forward, adjoint: compute_gradient(solver, forward, adjoint)
get_state_fn = lambda: np.array(solver.diffusion.spatial_angle).ravel()
set_state_fn = lambda state: solver.diffusion.update(
    {'spatial_angle': xr.DataArray(state.reshape(*solver.diffusion.spatial_angle.shape), dims=['x', 'y'])}
)

forward_op = ForwardOperator(
    forward_fn=forward_fn,
    adjoint_fn=adjoint_fn,
    gradient_fn=gradient_fn,
    set_state_fn=set_state_fn,
    get_state_fn=get_state_fn
)

objective_fn = ObjectiveFunction.l2(measurements, forward_op)

# Define SummaryWriter and add measurement and ground-truth images
writer = SummaryWriter(logdir=os.path.join('runs/priniple_angle_known_noise_zero_init_seed{}'.format(solver.seed)))
writer.average_image('average_frame/measurements', measurements)
writer.diffusion('diffusion/true', diffusion_true)
writer.advection('advection/true', advection_true)

# Define callback functions for routine checks/analysis of the state
callback_fn = [
    CallbackFn(lambda: writer.add_scalar('Loss', objective_fn.loss, optimizer.iteration)),
    CallbackFn(lambda: writer.diffusion('diffusion/estimate', solver.diffusion, optimizer.iteration)),
    CallbackFn(lambda: writer.average_image('average_frame/estimate', solver.run_symmetric(verbose=False), optimizer.iteration), ckpt_period=10*60),
    CallbackFn(lambda: optimizer.save_checkpoint(solver, writer.logdir), ckpt_period=1*60*60)
]

# Initialize principle angle
initial_state = solver.diffusion.spatial_angle
initial_state = initial_state * (0.0 * initial_state.where(initial_state.r <= 0.5)).fillna(1.0)

options={'maxiter': 10000, 'maxls': 100, 'disp': True, 'gtol': 1e-16, 'ftol': 1e-16}
optimizer = Optimizer(objective_fn, callback_fn, options=options)

measurements.to_netcdf(os.path.join(writer.logdir, 'measurements.nc'))
optimizer.save_checkpoint(solver, writer.logdir, name='initial_state.nc')
result = optimizer.minimize(initial_state=np.array(initial_state).ravel())
optimizer.save_checkpoint(solver, writer.logdir, name='final_result.nc')
writer.close()
