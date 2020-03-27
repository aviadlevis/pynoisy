"""
TODO: Some documentation and general description goes here.
"""
import numpy as np
from scipy.optimize import minimize, OptimizeResult

import time
from pynoisy.forward import PDESolver, SamplerPDESolver
from pynoisy.advection import Advection
from pynoisy.diffusion import RingDiffusion
from collections import OrderedDict
import pickle
import warnings

class Estimator(object):
    def __init__(self):
        self._num_parameters = None
        self._min_bound = None
        self._max_bound = None

    def set_state(self):
        return

    def get_state(self):
        return

    def get_gradient(self, backprop, forward, diffusion, advection):
        return

    def get_bounds(self):
        return

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def min_bound(self):
        return self._min_bound

    @property
    def max_bound(self):
        return self._max_bound


class RingDiffusionAngleEstimator(RingDiffusion, Estimator):
    def __init__(self, ring_diffusion, dx=1e-2, min_bound=-np.pi/2, max_bound=np.pi/2):
        super().__init__(ring_diffusion.opening_angle, ring_diffusion.tau, ring_diffusion.lam,
                         ring_diffusion.scaling_radius,ring_diffusion.tensor_ratio)
        self._dx = dx
        self._min_bound = min_bound
        self._max_bound = max_bound
        self._num_parameters = 1

    def get_gradient(self, backprop, forward, diffusion, advection):
        diffusion_f = RingDiffusion(self.opening_angle + self.dx, self.tau, self.lam, self.scaling_radius, self.tensor_ratio)
        jacobian_source = (diffusion_f.get_laplacian(forward, advection.v) -
                           diffusion.get_laplacian(forward, advection.v)) / self.dx
        jacobian_source = np.flip(jacobian_source, axis=0)
        gradient = np.mean(jacobian_source * backprop)
        return gradient

    def set_state(self, state):
        """
        Set the estimator state.

        Parameters
        ----------
        state: np.array(dtype=np.float64)
            The state to set the estimator
        """
        super(RingDiffusionAngleEstimator, self).__init__(state, self.tau, self.lam, self.scaling_radius, self.tensor_ratio)

    def get_state(self):
        """
        Retrieve the medium state.

        Returns
        -------
        state: np.array(dtype=np.float64)
            The current state.
        """
        return self.opening_angle

    def get_bounds(self):
        """
        Retrieve the bounds for every parameter (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        return [(self.min_bound, self.max_bound)] * self.num_parameters

    @property
    def dx(self):
        return self._dx


class Optimizer(object):
    """
    The Optimizer class takes care of the under the hood of the optimization process.
    To run the optimization the following methods should be called:
       [required] optimizer.set_measurements()
       [required] optimizer.set_solver()
       [required] optimizer.set_medium_estimator()
       [optional] optimizer.set_writer()

    Parameters
    ----------
    n_jobs: int, default=1
        The number of jobs to divide the gradient computation into
    """
    def __init__(self, n_jobs=1):
        self._solver = None
        self._adjoint_solver = None
        self._measurements = None
        self._writer = None
        self._iteration = 0
        self._loss = None
        self._n_jobs = n_jobs
        self._estimators = OrderedDict()

    def set_measurements(self, measurements):
        """
        Set the measurements (data-fit constraints)

        Parameters
        ----------
        measurements: pynoisy.Movie
            A Movie object storing measured images
        """
        self._measurements = measurements

    def set_solver(self, solver):
        """
        Set the PDESolver for the iterations.

        Parameters
        ----------
        solver: pynoisy.PDESolver or pynoisy.SamplerPDESolver
            The PDESolver for the solution iterations
        """
        self._solver = solver
        self._adjoint_solver = PDESolver(-solver.advection, solver.diffusion, solver.forcing_strength)
        if isinstance(solver.diffusion, Estimator):
            self.estimators['diffusion'] = solver.diffusion
        if isinstance(solver.advection, Estimator):
            self.estimators['advection'] = solver.advection

    def set_writer(self, writer):
        """
        Set a log writer to upload summaries into tensorboard.

        Parameters
        ----------
        writer: pynoisy.SummaryWriter
            Wrapper for the tensorboardX summary writer.
        """
        self._writer = writer
        if writer is not None:
            self._writer.attach_optimizer(self)

    def objective_fun(self):
        """
        The objective function (cost) and gradient at the current state.

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels
        gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the objective function with respect to the state parameters

        Notes
        -----
        This function also saves the current synthetic images for visualization purpose
        """
        error = self.forward_pass()
        self._loss = np.abs(error).mean()
        backprop = self.adjoint_pass(error)
        gradients = [estimator.get_gradient(
            backprop.frames, self.measurements.frames, self.solver.diffusion, self.solver.advection)
            for estimator in self.estimators.values()]
        return self.loss, gradients

    def forward_pass(self):
        if isinstance(self.solver, SamplerPDESolver):
            synthetic_measurements = self.solver.run(
                evolution_length=self.measurements.duration, n_jobs=self.n_jobs, verbose=False
            )
            error = (synthetic_measurements.mean().frames - self.measurements.frames) / \
                    synthetic_measurements.std().frames

        elif isinstance(self.solver, PDESolver):
            synthetic_measurements = self.solver.run(
                evolution_length=self.measurements.duration,
                verbose=False
            )
            error = synthetic_measurements.frames - self.measurements.frames

        return error

    def adjoint_pass(self, error):
        backprop = self.adjoint_solver.run_adjoint(
                source=error,
                evolution_length=self.measurements.duration,
                verbose=False
            )
        return backprop

    def callback(self):
        """
        The callback function invokes the callbacks defined by the writer (if any).
        Additionally it keeps track of the iteration number.
        """
        self._iteration += 1

        # Writer callback functions
        if self.writer is not None:
            for callbackfn, kwargs in zip(self.writer.callback_fns, self.writer.kwargs):
                time_passed = time.time() - kwargs['ckpt_time']
                if time_passed > kwargs['ckpt_period']:
                    kwargs['ckpt_time'] = time.time()
                    callbackfn(kwargs)

    def get_num_parameters(self):
        """
        get the number of parameters to be estimated by accumulating all the internal estimator parameters.

        Returns
        -------
        num_parameters: int
            The number of parameters to be estimated.
        """
        num_parameters = []
        for estimator in self.estimators.values():
            num_parameters.append(estimator.num_parameters)
        return num_parameters

    def get_bounds(self):
        """
        Retrieve the bounds for every parameter from the MediumEstimator (used by scipy.minimize)

        Returns
        -------
        bounds: list of tuples
            The lower and upper bound of each parameter
        """
        bounds = []
        for estimator in self.estimators.values():
            bounds.extend(estimator.get_bounds())
        return bounds

    def get_state(self):
        """
        Retrieve MediumEstimator state

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        return [estimator.get_state() for estimator in self.estimators.values()]

    def set_state(self, state):
        """
        Set the state of the optimization. This means:
          1. Setting the MediumEstimator state
          2. Updating the RteSolver medium
          3. Computing the direct solar flux
          4. Computing the current RTE solution with the previous solution as an initialization

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        for (name, estimator), state in zip(self.estimators.items(), state):
            estimator.set_state(state)
            if name == 'diffusion':
                self.solver.set_diffusion(estimator)
                self.adjoint_solver.set_diffusion(estimator)
            if name == 'advection':
                self.solver.set_advection(estimator)
                self.adjoint_solver.set_advection(-estimator)

    def save_state(self, path):
        """
        Save Optimizer state to file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'wb')
        file.write(pickle.dumps(self.get_state(), -1))
        file.close()

    def load_state(self, path):
        """
        Load Optimizer from file.

        Parameters
        ----------
        path: str,
            Full path to file.
        """
        file = open(path, 'rb')
        data = file.read()
        file.close()
        state = pickle.loads(data)
        self.set_state(state)


    @property
    def solver(self):
        return self._solver

    @property
    def adjoint_solver(self):
        return self._adjoint_solver

    @property
    def measurements(self):
        return self._measurements

    @property
    def num_parameters(self):
        return self._num_parameters

    @property
    def writer(self):
        return self._writer

    @property
    def iteration(self):
        return self._iteration

    @property
    def n_jobs(self):
        return self._n_jobs

    @property
    def loss(self):
        return self._loss

    @property
    def estimators(self):
        return self._estimators


class LBFGSbOptimizer(Optimizer):
    """
    L-BFGS-B optimization method.

    Parameters
    ----------
    n_jobs: int, default=1
        The number of jobs to divide the gradient computation into

    Notes
    -----
    For documentation:
        https://docs.scipy.org/doc/scipy/reference/optimize.minimize-LBFGSbOptimizerOptimizer.html

    """
    def __init__(self, options={'maxiter': 100, 'maxls': 100, 'disp': True, 'gtol': 1e-16, 'ftol': 1e-16}, n_jobs=1):
        super(self, LBFGSbOptimizer).__init__(n_jobs)
        self._options = options

    def callback(self, state):
        """
        The callback function invokes the callbacks defined by the writer (if any).
        Additionally it keeps track of the iteration number.
        """
        super(LBFGSbOptimizer, self).callback()

    def objective_fun(self, state):
        """
        The objective function (cost) and gradient at the current state.

        Parameters
        ----------
        state: np.array(shape=(self.num_parameters, dtype=np.float64)
            The current state vector

        Returns
        -------
        loss: np.float64
            The total loss accumulated over all pixels
        gradient: np.array(shape=(self.num_parameters), dtype=np.float64)
            The gradient of the objective function with respect to the state parameters

        Notes
        -----
        This function also saves the current synthetic images for visualization purpose
        """
        self.set_state(state)
        loss, gradient = super(LBFGSbOptimizer, self).objective_fun()
        return np.array(loss), np.array(gradient)

    def minimize(self):
        """
        Local minimization with respect to the parameters defined.
        """
        self._num_parameters = self.get_num_parameters()
        result = minimize(fun=self.objective_fun,
                          x0=self.get_state(),
                          method='L-BFGS-B',
                          jac=True,
                          bounds=self.get_bounds(),
                          options=self.options,
                          callback=self.callback)
        return result

    def get_state(self):
        """
        Retrieve MediumEstimator state

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        return np.array(super(LBFGSbOptimizer, self).get_state())

    def set_state(self, state):
        """
        Set the state of the optimization. This means:
          1. Setting the MediumEstimator state
          2. Updating the RteSolver medium
          3. Computing the direct solar flux
          4. Computing the current RTE solution with the previous solution as an initialization

        Returns
        -------
        state: np.array(dtype=np.float64)
            The state of the medium estimator
        """
        states = np.split(state, np.cumsum(self.num_parameters[:-1]))
        super(LBFGSbOptimizer, self).set_state(states)


class AdamOptimizer(Optimizer):
    """Stochastic gradient descent optimizer with Adam

    Parameters
    ----------
    options = {
        'lr_init': float, default=0.001
            The initial learning rate used. It controls the step-size in updating
            the weights
        'beta1': float, default=0.9
            Exponential decay rate for estimates of first moment vector, should be
            in [0, 1)
        'beta2': float, default=0.999
            Exponential decay rate for estimates of second moment vector, should be
            in [0, 1)
        'eps': float, default=1e-8
            Value for numerical stability
    }
    n_jobs: int, default=1
        The number of jobs to divide the gradient computation into

    Notes
    -----
    This code was taken from scikit-learn GitHub:
        https://github.com/scikit-learn/scikit-learn/blob/2f992722fd4bda6d31d7dabfa1f5f55261b241e5/sklearn/neural_network/_stochastic_optimizers.py)
    All default values are from the original Adam paper: Kingma, Diederik, and Jimmy Ba.
        "Adam: A method for stochastic optimization." arXiv preprint arXiv:1412.6980 (2014).
    """

    def __init__(self, options={'lr_init': 0.1, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8}, n_jobs=1):
        super().__init__(n_jobs)
        self.learning_rate_init = options['lr_init']
        self.beta_1 = options['beta1']
        self.beta_2 = options['beta2']
        self.epsilon = options['eps']
        self.t = 0

    def callback(self):
        self._iteration += 1
        print('Iter: {}      Angle: {}       Loss:{}     Best (Angle,Loss):({},{})'.format(
            self.iteration, self.get_state(), self.loss, self.best_state, self.best_loss))

    def minimize(self, maxiter=1000, tol=1e-5, n_iter_no_change=20):
        """

        Parameters
        ----------
        maxiter : int, default=200
            Maximum number of iterations. The solver iterates until convergence
            (determined by 'tol') or this number of iterations. For stochastic
            solvers ('sgd', 'adam'), note that this determines the number of epochs
            (how many times each data point will be used), not the number of
            gradient steps.

        tol : float, default=1e-4
            Tolerance for the optimization. When the loss or score is not improving
            by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
            unless ``learning_rate`` is set to 'adaptive', convergence is
            considered to be reached and training stops.

        n_iter_no_change : int, default=10
            Maximum number of epochs to not meet ``tol`` improvement.
            Only effective when solver='sgd' or 'adam'
        """
        self._num_parameters = self.get_num_parameters()
        self.maxiter = maxiter
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.ms = [np.zeros_like(param) for param in self.estimators.values()]
        self.vs = [np.zeros_like(param) for param in self.estimators.values()]
        self.best_loss = np.Inf
        self.no_improvement_count = 0
        self.best_state = self.get_state()
        early_stopping = False

        for iter in range(maxiter):
            loss, gradients = super(AdamOptimizer, self).objective_fun()
            updates = self.get_updates(gradients)
            new_state = [state + update for state, update in zip(self.get_state(), updates)]
            self.set_state(new_state)
            self.update_no_improvement_count()

            # Execute callback function
            self.callback()

            if self.no_improvement_count > self.n_iter_no_change:
                early_stopping = True
                print("Loss did not improve more than tol=%f"
                      " for %d consecutive iterations. Stopping" % (self.tol, self.n_iter_no_change))
                break

            if self.iteration == self.maxiter:
                warnings.warn("Stochastic Optimizer: Maximum iterations (%d) " 
                              "reached and the optimization hasn't converged yet." % self.maxiter)

        if early_stopping:
            self.state = self.best_state

        return self.state

    def update_no_improvement_count(self):
        if self.loss > self.best_loss - self.tol:
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
        if self.loss < self.best_loss:
            self.best_loss = self.loss
            self.best_state = self.get_state()

    def get_updates(self, grads):
        """Get the values used to update params with given gradients
        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates