"""
TODO: Some documentation and general description goes here.
"""
import numpy as np
import time
import tensorboardX
import matplotlib.pyplot as plt
import os
import xarray as xr
from matplotlib.colors import Normalize
import pynoisy.utils

class SummaryWriter(tensorboardX.SummaryWriter):
    def __init__(self, logdir=None, comment='', purge_step=None, max_queue=10, flush_secs=120,
                 filename_suffix='', write_to_disk=True, log_dir=None, **kwargs):
        super().__init__(logdir, comment, purge_step, max_queue, flush_secs,
                         filename_suffix, write_to_disk, log_dir, **kwargs)

    def spatial_angle(self, tag, spatial_angle, mask=None, global_step=None):
        mask = np.ones_like(spatial_angle) if mask is None else mask
        np.mod(spatial_angle, np.pi).where(mask).plot(cmap='hsv')
        self.add_figure(tag, plt.gcf(), global_step)


    def envelope(self, tag, envelope, global_step=None):
        plt.style.use('default')
        envelope.plot()
        self.add_figure(tag, plt.gcf(), global_step)


    def diffusion(self, tag, diffusion, mask=None, global_step=None, vector_plot=False, envelope=None, diffusion_coef=True):
        """TODO"""
        plt.style.use('default')

        if vector_plot:
            diffusion.noisy_methods.plot_principal_axis()
            self.add_figure(tag + '_principle', plt.gcf(), global_step)
            diffusion.noisy_methods.plot_secondary_axis()
            self.add_figure(tag + '_secondary', plt.gcf(), global_step)

        if envelope is None:
            np.mod(diffusion.spatial_angle, np.pi).where(mask).plot(cmap='hsv')
        else:
            alphas = Normalize(0, envelope.max(), clip=True)(envelope)
            colors = Normalize(0, np.pi)(np.mod(diffusion.spatial_angle, np.pi))
            cmap = plt.cm.hsv
            colors = cmap(colors)
            colors[..., -1] = alphas
            plt.imshow(colors)
            plt.gca().invert_yaxis()

        self.add_figure(tag + '_angle', plt.gcf(), global_step)
        diffusion.correlation_length.where(mask).plot()
        self.add_figure(tag + '_correlation_length', plt.gcf(), global_step)
        diffusion.correlation_time.where(mask).plot()
        self.add_figure(tag + '_correlation_time', plt.gcf(), global_step)
        if diffusion_coef:
            diffusion.diffusion_coefficient.where(mask).plot()
            self.add_figure(tag + '_diffusion_coefficient', plt.gcf(), global_step)

    def advection(self, tag, advection, global_step=None):
        plt.style.use('default')
        advection.noisy_methods.plot_velocity()
        self.add_figure(tag, plt.gcf(), global_step)

    def average_image(self, tag, video, global_step=None):
        plt.style.use('default')
        video.mean('t').plot()
        self.add_figure(tag, plt.gcf(), global_step)

class ForwardOperator(object):
    def __init__(self, forward_fn, adjoint_fn, gradient_fn, set_state_fn, get_state_fn):
        self._forward_fn = forward_fn
        self._gradient_fn = gradient_fn
        self._adjoint_fn = adjoint_fn
        self._set_state_fn = set_state_fn
        self._get_state_fn = get_state_fn

    def __call__(self, state):
        self.set_state(state)
        return self._forward_fn()

    def adjoint(self, source):
        return self._adjoint_fn(source)

    def gradient(self, forward, adjoint):
        return self._gradient_fn(forward, adjoint)

    def set_state(self, state):
        self._set_state_fn(state)

    def get_state(self):
        return self._get_state_fn()

    @classmethod
    def krylov(cls, forward_fn, adjoint_fn, gradient_fn, set_state_fn, get_state_fn, measurements,
               degree=8, k_matrix_fn=None):
        def krylov_fn():
            k_matrix = pynoisy.utils.get_krylov_matrix(measurements, forward_fn, degree) if k_matrix_fn is None \
                else k_matrix_fn(measurements, degree)
            coefs = np.linalg.lstsq(k_matrix.T, np.array(measurements).ravel(), rcond=None)[0]
            rec = np.dot(coefs.T, k_matrix).reshape(*measurements.shape)
            return xr.DataArray(rec, coords=measurements.coords)

        return cls(lambda: krylov_fn(), adjoint_fn, gradient_fn, set_state_fn, get_state_fn)

    @property
    def min_bound(self):
        return self._min_bound

    @property
    def max_bound(self):
        return self._max_bound

class ObjectiveFunction(object):
    def __init__(self, measurements, forward_op, loss_fn, min_bounds=None, max_bounds=None):
        self.set_measurements(measurements)
        self._loss_fn = loss_fn
        self._forward_op = forward_op
        self._loss = None
        min_bounds, max_bounds = np.broadcast_arrays(min_bounds, max_bounds, forward_op.get_state())[:2]
        self._bounds = list(zip(np.atleast_1d(min_bounds), np.atleast_1d(max_bounds)))

    def __call__(self, state):
        loss, gradient = self._loss_fn(state, self.forward_op, self.measurements)
        self._loss = loss
        return np.array(loss), np.array(gradient)

    def set_measurements(self, measurements):
        self._measurements = measurements

    @classmethod
    def l2(cls, measurements, forward_op, min_bounds=None, max_bounds=None):
        def l2_loss_fn(state, forward_op, measurements):
            synthetic_movie = forward_op(state)
            error = synthetic_movie - measurements
            loss = (error ** 2).mean()
            adjoint = forward_op.adjoint(error)
            gradient = forward_op.gradient(synthetic_movie, adjoint)
            return np.array(loss), np.array(gradient).ravel()

        return cls(measurements, forward_op, l2_loss_fn, min_bounds, max_bounds)

    @property
    def measurements(self):
        return self._measurements

    @property
    def forward_op(self):
        return self._forward_op

    @property
    def bounds(self):
        return self._bounds

    @property
    def loss(self):
        return self._loss

class PriorFunction(object):
    def __init__(self, prior_fn, scale=1.0):
        self._prior_fn = prior_fn
        self._loss = None
        self._scale = scale

    def __call__(self, state):
        loss, gradient = self._prior_fn(state)
        self._loss = self.scale * loss
        return np.array(self._loss), self.scale * np.array(gradient)

    @classmethod
    def mahalanobis(cls, mean=0, cov=None, cov_inverse=None, scale=1.0e-4):
        cov_inverse = np.linalg.inv(cov) if cov is not None else cov_inverse
        def mahalanobis_loss_fn(state):
            gradient = np.matmul(cov_inverse, state-mean)
            loss = np.matmul(state-mean, gradient)
            return np.array(loss), 2*np.array(gradient).ravel()
        return cls(prior_fn=mahalanobis_loss_fn, scale=scale)

    @property
    def loss(self):
        return self._loss

    @property
    def scale(self):
        return self._scale

class Optimizer(object):
    """
    Optmizer wrapps the scipy optimization methods.

    Notes
    -----
    For documentation:
        https://docs.scipy.org/doc/scipy/reference/optimize.html

    """

    def __init__(self,
                 objective_fn,
                 prior_fn=None,
                 callback_fn=None,
                 method='L-BFGS-B',
                 options={'maxiter': 100, 'maxls': 100, 'disp': True, 'gtol': 1e-16, 'ftol': 1e-16}
                 ):
        from scipy.optimize import minimize
        self._minimize = minimize
        self._method = method
        self._options = options
        self._objective_fn = objective_fn
        self._prior_fn = np.atleast_1d(prior_fn) if prior_fn is not None else None
        self._callback_fn = np.atleast_1d(callback_fn)
        self._callback = None if callback_fn is None else self.callback

    def callback(self, state):
        """
        The callback function invokes the callbacks defined by the writer (if any).
        Additionally it keeps track of the iteration number.
        """
        self._iteration += 1
        [function() for function in self._callback_fn]

    def objective(self, state):
        """
        The callback function invokes the callbacks defined by the writer (if any).
        Additionally it keeps track of the iteration number.
        """
        loss, gradient = self._objective_fn(state)
        if self._prior_fn is not None:
            p_loss, p_gradient = np.array([prior(state) for prior in self._prior_fn]).sum(axis=0)
            loss, gradient = loss + p_loss, gradient + p_gradient
        return loss, gradient

    def minimize(self, initial_state, iteration_step=0):
        """
        Local minimization with respect to the parameters defined.
        """
        self._iteration = iteration_step
        args = {
            'fun': self.objective,
            'x0': initial_state,
            'method': self.method,
            'jac': True,
            'options': self.options,
            'callback': self._callback
        }

        if self.method not in ['CG', 'Newton-CG']:
            args['bounds'] = self.objective_fn.bounds
        result = self._minimize(**args)
        return result

    def save_checkpoint(self, solver, dir, name=None):
        if not os.path.exists(dir):
            os.makedirs(dir)
        name = 'checkpoint{:03d}.nc'.format(self.iteration) if name is None else name
        path = os.path.join(dir, name)
        print('[{}] Saving solver state to NetCDF: {}'.format(time.strftime("%d-%b-%Y-%H:%M:%S"), path))
        solver.save(path)

    @property
    def objective_fn(self):
        return self._objective_fn

    @property
    def iteration(self):
        return self._iteration

    @property
    def method(self):
        return self._method

    @property
    def options(self):
        return self._options

class CallbackFn(object):
    def __init__(self, callback_fn, ckpt_period=-1):
        self.ckpt_period = ckpt_period
        self.ckpt_time = time.time()
        self._callback_fn = callback_fn

    def __call__(self):
        time_passed = time.time() - self.ckpt_time
        if time_passed > self.ckpt_period:
            self.ckpt_time = time.time()
            self._callback_fn()


def spatial_angle_gradient(solver, dx=1e-5):
    """TODO"""
    def gradient(solver, forward, adjoint, mask, dx):
        spatial_angle = solver.diffusion.spatial_angle.copy()
        source = solver.get_laplacian(forward)
        gradient = np.zeros(shape=solver.params.num_unknowns)
        for n, (i, j) in enumerate(zip(*np.where(mask))):
            solver.diffusion.spatial_angle[i, j] = spatial_angle[i, j] + dx
            source_ij = solver.get_laplacian(forward) - source
            solver.diffusion.spatial_angle[i, j] = spatial_angle[i, j]
            source_ij = source_ij / dx
            gradient[n] += (adjoint * source_ij).mean()
        return gradient

    mask = solver.params.mask if hasattr(solver.params, 'mask') else xr.ones_like(solver.diffusion.spatial_angle, dtype=np.bool)
    gradient_fn = lambda forward, adjoint: gradient(solver, forward, adjoint, mask, dx)
    return gradient_fn

def spatial_angle_set_state(solver):
    """TODO"""
    def set_state(solver, state, mask):
        solver.diffusion.spatial_angle.values[mask] = state
    mask = solver.params.mask if hasattr(solver.params, 'mask') else xr.ones_like(solver.diffusion.spatial_angle, dtype=np.bool)
    set_state_fn = lambda state: set_state(solver, state, mask)
    return set_state_fn

def spatial_angle_get_state(solver):
    """TODO"""
    mask = solver.params.mask if hasattr(solver.params, 'mask') else xr.ones_like(solver.diffusion.spatial_angle, dtype=np.bool)
    get_state_fn = lambda: solver.diffusion.spatial_angle.values[mask]
    return get_state_fn

