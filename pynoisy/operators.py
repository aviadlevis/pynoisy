"""
LinearOperators [1,2] objects and containers used for optimization [3].
Operators should implement the methods: `_matvec(self, x)` and `_rmatvec(self, x)`. See [1] for more information.

Notes
-----
To check the validity of an adjoint implementation with respect to the forward use: `pynoisy.operators.dottest`

References
----------
[1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html
[2] https://pylops.readthedocs.io/en/latest/
[3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
"""
import numpy as _np
import xarray as _xr
from scipy.sparse.linalg import LinearOperator as _LinearOperator

def dottest(Op, tol=1e-6, complexflag=0, raiseerror=True, verb=False):
    """Dot test.
    Generate random vectors :math:`\mathbf{u}` and :math:`\mathbf{v}` and perform dot-test to verify the validity of
    forward and adjoint operators. This test can help to detect errors in the operator implementation.
    This function was taken from PyLops [1].

    Parameters
    ----------
    Op : :obj:`pylops.LinearOperator`
        Linear operator to test.
    tol : :obj:`float`, optional
        Dottest tolerance
    complexflag : :obj:`bool`, optional
        generate random vectors with real (0) or complex numbers
        (1: only model, 2: only data, 3:both)
    raiseerror : :obj:`bool`, optional
        Raise error or simply return ``False`` when dottest fails
    verb : :obj:`bool`, optional
        Verbosity

    Raises
    ------
    ValueError
        If dot-test is not verified within chosen tolerance.

    Notes
    -----
    A dot-test is mathematical tool used in the development of numerical
    linear operators.
    More specifically, a correct implementation of forward and adjoint for
    a linear operator should verify the following *equality*
    within a numerical tolerance:
    .. math::
        (\mathbf{Op}*\mathbf{u})^H*\mathbf{v} =
        \mathbf{u}^H*(\mathbf{Op}^H*\mathbf{v})

    References
    ----------
    [1] https://github.com/PyLops/pylops/blob/master/pylops/utils/dottest.py
    """
    nr, nc = Op.shape

    if complexflag in (0, 2):
        u = _np.random.randn(nc)
    else:
        u = _np.random.randn(nc) + 1j*_np.random.randn(nc)

    if complexflag in (0, 1):
        v = _np.random.randn(nr)
    else:
        v = _np.random.randn(nr) + 1j*_np.random.randn(nr)

    y = Op.matvec(u)   # Op * u
    x = Op.rmatvec(v)  # Op'* v

    if complexflag == 0:
        yy = _np.dot(y, v) # (Op  * u)' * v
        xx = _np.dot(u, x) # u' * (Op' * v)
    else:
        yy = _np.vdot(y, v) # (Op  * u)' * v
        xx = _np.vdot(u, x) # u' * (Op' * v)

    # evaluate if dot test is passed
    if complexflag == 0:
        if _np.abs((yy - xx) / ((yy + xx + 1e-15) / 2)) < tol:
            if verb: print('Dot test passed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                           % (yy, xx))
            return True
        else:
            if raiseerror:
                raise ValueError('Dot test failed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                                 % (yy, xx))
            if verb: print('Dot test failed, v^T(Opu)=%f - u^T(Op^Tv)=%f'
                           % (yy, xx))
            return False
    else:
        checkreal = _np.abs((_np.real(yy) - _np.real(xx)) /
                           ((_np.real(yy) + _np.real(xx)+1e-15) / 2)) < tol
        checkimag = _np.abs((_np.real(yy) - _np.real(xx)) /
                           ((_np.real(yy) + _np.real(xx)+1e-15) / 2)) < tol
        if checkreal and checkimag:
            if verb:
                print('Dot test passed, v^T(Opu)=%f%+fi - u^T(Op^Tv)=%f%+fi'
                      % (yy.real, yy.imag, xx.real, xx.imag))
            return True
        else:
            if raiseerror:
                raise ValueError('Dot test failed, v^H(Opu)=%f%+fi '
                                 '- u^H(Op^Hv)=%f%+fi'
                                 % (yy.real, yy.imag, xx.real, xx.imag))
            if verb:
                print('Dot test failed, v^H(Opu)=%f%+fi - u^H(Op^Hv)=%f%+fi'
                      % (yy.real, yy.imag, xx.real, xx.imag))
            return False

class ModulateOp(_LinearOperator):
    """
    Envelope modulation Operator in the form of scipy.sparse.linalg.LinearOperator object.
    The forward call is equivalent to pynoisy.forward.modulate().

    Parameters
    ----------
    grf: xr.DataArray
        Output DataArray GRF with dims ['t', 'y', 'x'].
    alpha: float, default=1.0
        Exponential rate
    dtype: datatype, default=np.float64
    """
    def __init__(self, grf, alpha=1.0, dtype=_np.float64):
        self.modulation = _np.exp(alpha * grf).data.reshape(grf['t'].size, -1)

        # Shape and datatype
        nt, ny, nx = grf['t'].size, grf['y'].size, grf['x'].size
        self._nt = nt
        self.shape = (nt * ny * nx, ny * nx)
        self.dtype = dtype

    def _matvec(self, x):
        output = (self.modulation * x).ravel()
        return output

    def _rmatvec(self, x):
        return _np.sum(_np.split(self.modulation.ravel() * x, self._nt), axis=0)

class ObserveOp(_LinearOperator):
    """
    An EHT observation operator in the form of scipy.sparse.linalg.LinearOperator object.
    The forward call is equivalent to xarray.utils_observe.block_observe_same_nonoise().

    Parameters
    ----------
    obs; ehtim.Observation,
        ehtim Observation object.
    movie_coords: xr.Coordinates,
        The coordinates of the movie
    dtype: np.dtype, default=np.complex128,
        Datatype
    """
    def __init__(self, obs, movie_coords, dtype=_np.complex128):

        import ehtim.observing.obs_helpers as _obsh

        # Forward coordinates
        obslist = obs.tlist()
        u_list = [obsdata['u'] for obsdata in obslist]
        v_list = [obsdata['v'] for obsdata in obslist]
        t_list = [obsdata[0]['time'] for obsdata in obslist]
        t = _np.concatenate([obsdata['time'] for obsdata in obslist])
        u, v = _np.concatenate(u_list), _np.concatenate(v_list)
        self._vis_coords = {'t': ('index', t), 'u': ('index', u), 'v': ('index', v),
                            'uvdist': ('index', _np.sqrt(u ** 2 + v ** 2))}
        self._obstimes = t_list
        uv_per_t = _np.array([len(obsdata['v']) for obsdata in obslist])
        self._uvsplit = _np.cumsum(uv_per_t)[:-1]

        # Adjoint coordinates
        self._movie_coords = movie_coords

        # Define forward operator as a sequence (list) of matrix operations
        A = []
        self._nx = movie_coords['x'].size
        self._ny = movie_coords['y'].size
        self._nt = movie_coords['t'].size
        psize = movie_coords.to_dataset().utils_image.psize
        for ui, vi in zip(u_list, v_list):
            A.append(_obsh.ftmatrix(psize, self._nx, self._ny, _np.vstack((ui, vi)).T))
        self._A = A

        # Shape and datatype
        self.shape = (u.size, self._nt * self._ny * self._nx)
        self.dtype = dtype

    def _matvec(self, x):
        x = x.reshape(self._nt, self._nx, self._ny) if x.ndim != 3 else x
        x = _xr.DataArray(x, coords=self._movie_coords, dims=['t', 'y', 'x']).interp(
            t=self._obstimes, assume_sorted=True)
        output = _np.concatenate([
            _np.matmul(At, xt.data.ravel()) for At, xt in zip(self._A, x)
        ])
        return output

    def _rmatvec(self, x):
        x_list = _np.split(x, self._uvsplit)
        return _np.concatenate([_np.matmul(At.conj().T, xt) for At, xt in zip(self._A, x_list)])

class Loss(object):
    """
    A Loss container which aggregates data-fit and regularization Operators.
    This container is meant to be used together with scipy.optimize.minimize(fun=loss, jac=loss.jac).

    Parameters
    ----------
    datafit_ops: ,
        Data-fit operators which implemnt
    """
    def __init__(self, datafit_ops, datafit_weights, regularize_ops, regularize_weights):
        self.datafit_ops = _np.atleast_1d(datafit_ops)
        self.datafit_w = _np.atleast_1d(datafit_weights)
        if len(self.datafit_ops) != len(self.datafit_w):
            raise AttributeError('len(datafit_ops) != len(datafit_weights)')

        self.regularize_ops = _np.atleast_1d(regularize_ops)
        self.regularize_w = _np.atleast_1d(regularize_weights)
        if len(self.regularize_ops) != len(self.regularize_w):
            raise AttributeError('len(regularize_ops) != len(regularize_weights)')

    def __call__(self, x):
        datafit_loss = _np.sum([w * data_op(x) for data_op, w in zip(self.datafit_ops, self.datafit_w)])
        regularize_loss = _np.sum([w * reg_op(x) for reg_op, w in zip(self.regularize_ops, self.regularize_w)])
        return datafit_loss + regularize_loss

    def jac(self, x):
        datafit_grad = _np.sum([w * data_op.gradient(x) for data_op, w in zip(self.datafit_ops, self.datafit_w)], axis=0)
        regularize_grad = _np.sum([w * reg_op.gradient(x) for reg_op, w in zip(self.regularize_ops, self.regularize_w)],
                                 axis=0)
        return (datafit_grad + regularize_grad).real.astype(_np.float64)

class LossOperator(object):
    """
    A LossOperator container which is inherited by the specific loss implementation.
    LossOperators should implement the methods: `__call__(self, x)` and `gradient(self, x)`.
    """
    def __init__(self):
        pass
    def __call__(self, x):
        pass
    def gradient(self, x):
        pass

class L2LossOp(LossOperator):
    """
    An l2 LossOperator implementing the computation and gradient of ||measurements - forwardOp(x)||^2

    Parameters
    ----------
    measurements: np.array,
        A 1D numpy array with measurement values.
    forwardOp: LinearOperator,
        A LinearOperator which implements: `_matvec(self, x)` and `_rmatvec(self, x)` (see [1])

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html
    """
    def __init__(self, measurements, forwardOp):
        self.measurements = measurements
        self.forwardOp = forwardOp

    def __call__(self, x):
        return _np.sum(_np.abs(self.measurements - self.forwardOp * x) ** 2)

    def gradient(self, x):
        return self.forwardOp.H * (self.forwardOp * x - self.measurements)

class MEMRegOp(LossOperator):
    """
    Maximum Entropy Method regularization LossOperator.
     Entropy(x; prior) = sum( x * log( x/(prior + eps) ) )

    Parameters
    ----------
    prior: np.array,
        A 1D numpy array which represents the (raveled) prior vector.
    eps: float, default=1e-5,
        A regularization parameter to avoid division by zero.
    """
    def __init__(self, prior, eps=1e-5):
        self.eps = eps
        self.prior = prior

    def __call__(self, x):
        return -_np.sum(x * _np.log((x + self.eps) / (self.prior + self.eps)))

    def gradient(self, x):
        return -_np.log((x + self.eps) / (self.prior + self.eps)) - 1

class FluxRegOp(LossOperator):
    """
    Total flux regularization LossOperator.

    Parameters
    ----------
    prior: float,
        The prior on the total flux.
    """
    def __init__(self, prior):
        self.prior = prior

    def __call__(self, x):
        return (_np.sum(x) - self.prior) ** 2

    def gradient(self, x):
        return 2 * (_np.sum(x) - self.prior) * _np.ones(len(x), dtype=_np.float64)