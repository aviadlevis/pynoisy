"""
Linear algebra utility functions to compute modes of the underlying stochastic partial differential equation (SPDE)
and projections onto subspaces spanned by those modes.

References
----------
..[1] Halko, N., Martinsson, P.G. and Tropp, J.A.. Finding structure with randomness: Probabilistic algorithms
      for constructing approximate matrix decompositions. SIAM review, 53(2), pp.217-288. 2011.
      url: https://epubs.siam.org/doi/abs/10.1137/090771806
"""
import xarray as xr
import numpy as np
import scipy.linalg
from scipy.sparse.linalg import lsqr
from tqdm.auto import tqdm

def basis_to_xarray(basis, dims, coords, attrs=None, name=None):
    """
    Return an xr.DataArray object form a numpy array basis.

    Parameters
    ----------
    basis: np.array,
        A 2D numpy array with the axis representing (features, vectors)
    dims: tuple,
        A tuple of dimensions
    coords: xr.Coordinates
        DataArray coordinates
    attrs: dict, optional
        DataArray attributes
    name: str, optional
        DataArray name
    """
    shape = [coords[dim].size for dim in dims]
    array = xr.DataArray(data=basis.T.reshape(shape), dims=dims, coords=coords, attrs=attrs, name=name)
    return array

def orthogonal_projection(vectors, subspace):
    """
    Project vectors onto an orthogonal subspace.

    Parameters
    ----------
    vectors: np.array,
        A 2D numpy array with the axis representing (features, vectors)
    subspace: tuple,
        A 2D numpy array with the axis representing (features, vectors)

    Returns
    -------
    projection: np.array
        An array with the vectors projection onto the subspace
    """
    return np.matmul(subspace, np.matmul(subspace.T, vectors))

def rsi_iteration(solver, input_vectors, orthogonal_subspace=None, n_jobs=4):
    """
    A single iteration of Randomized Subspace Iteration (RSI).

    Parameters
    ----------
    solver: pynoisy.forward.HGRFSolver,
        A Solver object which symbolically preforms matrix multiplication by solving the underlying SPDE
        See pynoisy/forward.py for more information.
    input_vectors: xr.DataArray,
        A DataArray with the input vectors along dimension 'sample'.
    test_vectors: xr.DataArray,
        A DataArray with the test vectors along dimension 'sample'. Used for convergence statistics.
    orthogonal_subspace: np.array, optional,
        A 2D array with subspace to orthogonalize against (e.g. for deflation). The shape is (features, vectors).
    n_jobs: int, default 4
        Number of parallel jobs for MPI solvers. See pynoisy.forward.HGRFSolver for details.

    Returns
    -------
    basis_xr: xr.DataArray
        A DataArray with the basis vectors after a single application of the forward operator.
        Note that these vectors are not orthogonal.
    """
    # Create an orthogonal basis from the input_vectors with QR decomposition
    input_vectors = input_vectors.linalg.qr_orthogonalization('sample')
    basis_xr = solver.run(source=input_vectors, n_jobs=n_jobs, verbose=0)

    # Orthogonalize against subspace with double MGS (Modified Grahm Schmidt)
    if orthogonal_subspace is not None:
        basis = basis_xr.linalg.to_basis(vector_dim='sample')
        basis = basis - orthogonal_projection(basis, orthogonal_subspace)
        basis = basis - orthogonal_projection(basis, orthogonal_subspace)
        basis_xr = basis_to_xarray(basis, basis_xr.dims, basis_xr.coords, basis_xr.attrs)

    return basis_xr

def randomized_subspace(solver, blocksize, maxiter, deflation_subspace=None, n_jobs=4, tol=1e-3, num_test_grfs=10,
                        verbose=True):
    """
    Compute top modes using Randomized Subspace Iteration (RSI) [1].

    Parameters
    ----------
    solver: pynoisy.forward.HGRFSolver,
        A Solver object which symbolically preforms matrix multiplication by solving the underlying SPDE
        See pynoisy/forward.py for more information.
    blocksize: int,
        The blocksize is approximately the size of the resulting subspace (minute ~2).
    maxiter: int,
        Maximum number of iterations for the RSI procedure.
    deflation_subspace: np.array, optional,
        A 2D array with subspace to orthogonalize against for deflation. The shape is (features, vectors).
    n_jobs: int, default 4
        Number of parallel jobs for MPI solvers. See pynoisy.forward.HGRFSolver for details.
    tol: float, default=1e-3
        Tolerance (slope) for convergence of the representation residual statistics. Used if num_test_grfs > 0.
    num_test_grfs: int, default=10,
        Number of test GRFs used to gather representation residual statistics.
        If set to zero than adaptive convergence criteria is ignored.
    verbose: bool, default=True
        Show tqdm progress bar and print convergence results.

    Returns
    -------
    modes: xr.Dataset
        A Dataset with the computed eigenvectors and eigenvalues as a function of 'degree'.
    residual: dict,
        residual['mean'] and residual['std'] contain a list of statistics for
        'num_test_grfs' representation errors as a function of iteration. If 'num_test_grfs'=0 then the lists are empty.

    References
    ----------
    ..[1] Halko, N., Martinsson, P.G. and Tropp, J.A.. Finding structure with randomness: Probabilistic algorithms
          for constructing approximate matrix decompositions. SIAM review, 53(2), pp.217-288. 2011.
          url: https://epubs.siam.org/doi/abs/10.1137/090771806
    """
    deflation_degree = 0
    if deflation_subspace is not None:
        if deflation_subspace.ndim > 2:
            raise AttributeError('deflation_subspace has dimensions greater than 2')
        deflation_degree = deflation_degree.shape[-1]

    # Generate test GRFs for convergence statistics
    if (num_test_grfs > 0):
        random_sources = solver.sample_source(num_samples=num_test_grfs)
        grf_fullrank = solver.run(source=random_sources, num_samples=num_test_grfs, verbose=0)

        # Transform to numpy arrays
        grf_fullrank = grf_fullrank.linalg.to_basis('sample')
        random_sources = random_sources.linalg.to_basis('sample')
        grf_mean_energy = np.mean(np.linalg.norm(grf_fullrank, axis=0) ** 2)

    solver.reseed(printval=False)
    basis = solver.sample_source(num_samples=blocksize)
    residual = {'mean': [], 'std': []}
    loop = tqdm(range(maxiter), desc='subspace iteration') if verbose else range(maxiter)
    for iter in loop:
        basis = rsi_iteration(solver, basis, deflation_subspace, n_jobs)

        # Adaptive convergence criteria based on test GRF statistics.
        if (num_test_grfs > 0):
            u, s, v = scipy.linalg.svd(basis.linalg.to_basis(vector_dim='sample'), full_matrices=False)
            grf_lowrank = np.matmul(u * s, np.matmul(u.T, random_sources))
            residual_samples = np.linalg.norm(grf_lowrank - grf_fullrank, axis=0) ** 2 / grf_mean_energy
            residual['mean'].append(np.mean(residual_samples))
            residual['std'].append(np.std(residual_samples))

            if (len(residual['mean']) > 2) and (np.abs(residual['mean'][-1] - residual['mean'][-2]) < tol) and \
                    (np.abs(residual['std'][-1] - residual['std'])[-2] < tol):
                if verbose:
                    print('RSI converged with residual_mean = {:1.3f} ; residual_std = {:1.3f}'.format(
                          residual['mean'][-1], residual['std'][-1]))
                break

    if (num_test_grfs == 0):
        u, s, v = scipy.linalg.svd(basis.linalg.to_basis(vector_dim='sample'), full_matrices=False)

    # Eigenvectors
    eigenvectors = basis_to_xarray(u, basis.dims, basis.coords, basis.attrs, name='eigenvectors')
    eigenvectors = eigenvectors.swap_dims({'sample': 'degree'}).drop('sample')

    # Eigenvalues
    degrees = range(deflation_degree, deflation_degree + basis.sample.size)
    eigenvalues = xr.DataArray(s, dims='degree', coords={'degree': degrees}, name='eigenvalues')

    modes = xr.merge([eigenvectors, eigenvalues])
    if (num_test_grfs > 0):
        residual_xr = xr.Dataset({'residual_mean': ('iteration', residual['mean']),
                                  'residual_std': ('iteration', residual['std'])})
        modes = xr.merge([modes, residual_xr])

    modes.attrs.update(
        blocksize=blocksize,
        maxiter=maxiter,
        tol=tol,
        num_test_grfs=num_test_grfs,
        iterations=iter+1,
        deflation_subspace_degree=deflation_degree
    )
    return modes

def projection_residual(vector, subspace, damp=0.0, return_projection=False, return_coefs=False):
    """
    Compute projection residual of the vector onto the subspace.
    The function solves ``min ||Ax - b||^2`` or the damped version: ``min ||Ax - b||^2 + d^2 ||x||^2``,
    where A is the subspace matrix, b is the input vector and d is the damping factor.

    Parameters
    ----------
    vector: xr.DataArray,
        An input DataArray to project onto the subspace.
    subspace: xr.DataArray,
        A DataArray with the spanning vectors along dimension 'degree'.
        Note that for low rank appoximation of a matrix the subspace should be the multiplication:
            eigenvectors * eigenvalues.
    damp: float, default=0.0
        Damping of the least-squares problem. This is a weight on the coefficients l2 norm: damp^2 * ||x||^2
    return_projection: bool, default=False,
        Return the projection as part of the output.
    return_coefs: bool, default=False,
        Return the projection coefficients (x) as part of the output.

    Returns
    -------
    output: tuple
        Output is: (residual, projection, coefficients) if return_projection=True and return_coefs=True.
            - residual is an xr.Dataset with 'data'=||Ax* - b||^2 and 'total'=||Ax* - b||^2 + d^2 ||x*||^2,
              where x* is the optimal x
            - projection is an xr.DataArray with the projected vector
            - coefficients are the projection coefficients as a function of 'degree'.

    Notes
    -----
    The hard work is done by the function 'lsqr_projection'.
    """
    projection_matrix = subspace.linalg.to_basis(vector_dim='degree')
    result = lsqr_projection(vector, projection_matrix, damp, return_projection, return_coefs)
    residual = xr.Dataset(data_vars={'data':result[0], 'total':result[1]}, attrs={'damp': damp})
    output = residual
    if return_projection:
        output = (residual, result[2])
    if return_coefs:
        coefs = xr.DataArray(result[-1], coords={'degree': subspace['degree']}, dims='degree')
        output = (residual, coefs) if return_projection is False else output + (coefs,)
    return output

def lsqr_projection(b, A, damp=0.0, return_projection=False, return_coefs=False, real_estimation=True):
    """
    Solve ``min ||Ax - b||^2`` or the damped version: ``min ||Ax - b||^2 + d^2 ||x||^2``,
    where A is the subspace matrix, b is the input vector and d is the damping factor.

    Parameters
    ----------
    b: xr.DataArray or np.array,
        An input DataArray or numpy array.
    A: np.array,
        A projection matrix with shape (features, vectors).
    damp: float, default=0.0
        Damping of the least-squares problem. This is a weight on the coefficients l2 norm: damp^2 * ||x||^2
    return_projection: bool, default=False,
        Return the projection as part of the output.
    return_coefs: bool, default=False,
        Return the projection coefficients (x) as part of the output.
    real_estimation: bool, default=True,
        True for solving only for real coefficient vector 'x'. This makes an impact if the measurements are complex.

    Returns
    -------
    output: tuple
        Output is: (r1, r2, projection, coefs) if return_projection=True and return_coefs=True.
            - r1 is the residual of the datafit: r1 = ||Ax* - b||^2, where x* is the optimal x
            - r2 is the total residual: r2=||Ax* - b||^2 + d^2 ||x*||^2, where x* is the optimal x
            - projection is an xr.DataArray (if b is a DataArray) with the projected vector
            - coefs is a numpy array with the projection coefficients
    """
    y_array = np.array(b).ravel()
    meas_length = b.size
    y_mask = np.isfinite(y_array)
    A_mask = np.isfinite(A)

    projection = np.full_like(y_array, fill_value=np.nan)

    size_diff = None
    if meas_length < A.shape[0]:
        y_array = np.concatenate((y_array, np.zeros(A.shape[0] - meas_length)))
        y_mask = np.concatenate((y_mask, np.ones(A.shape[0] - meas_length, dtype=np.bool)))
        size_diff = meas_length - A.shape[0]

    assert np.equal(A_mask, y_mask[:, None]).all(), "Masks of A matrix and b are not identical"

    A = A[A_mask].reshape(-1, A.shape[-1])
    y_array = y_array[y_mask]

    if (real_estimation) and b.dtype == 'complex':
        r2, coefs = lsqr_real(b, A, damp)
        projection[y_mask[:meas_length]] = np.dot(A[:size_diff], coefs)
        projection = projection.reshape(*b.shape)
        r1 = np.linalg.norm(b - projection) ** 2
    else:
        out = lsqr(A, y_array, damp=damp)
        coefs, r1, r2 = out[0], out[3]**2, out[4]**2

    output = (r1, r2)
    if return_projection:
        projection[y_mask[:meas_length]] = np.dot(A[:size_diff], coefs)
        projection = projection.reshape(*b.shape)
        if isinstance(b, xr.DataArray):
            projection = xr.DataArray(projection, coords=b.coords)
        output += (projection,)
    if return_coefs:
        output += (coefs,)
    return output

def lsqr_real(b, A, damp=0.0):
    """
    Solve ``min ||Ax - b||^2`` or the damped version: ``min ||Ax - b||^2 + d^2 ||x||^2``,
    where A is the subspace matrix, b is the input vector and d is the damping factor.

    The solution x* is the optimal *real* vector for the minimization defined above.

    Parameters
    ----------
    b: xr.DataArray or np.array,
        An input DataArray or numpy array.
    A: np.array,
        A projection matrix with shape (features, vectors).
    damp: float, default=0.0
        Damping of the least-squares problem. This is a weight on the coefficients l2 norm: damp^2 * ||x||^2

    Returns
    -------
    output: tuple,
        Output is: (fun, x): the residual and optimal *real-valued* solution x*.

    Notes
    -----
    The underlying optimization engine is scipy L-BFGS-B with mosly default parameters. See ref [1].

    References
    ----------
    [1] https://docs.scipy.org/doc/scipy-1.1.0/reference/optimize.minimize-lbfgsb.html
    """
    l2_loss = lambda x: np.linalg.norm(b - A.dot(x))**2 + np.linalg.norm(damp*x)**2
    l2_grad = lambda x: 2*np.ascontiguousarray(np.real(A.conj().T.dot(A.dot(x) - b)) - (damp**2) * x)
    out = scipy.optimize.minimize(l2_loss, jac=l2_grad, method='L-BFGS-B', x0=np.zeros(A.shape[-1], dtype=np.float64))
    return out['fun'], out['x']


@xr.register_dataarray_accessor("linalg")
class LinearAlgebraAccessor(object):
    """
    Register a custom accessor LinearAlgebraAccessor on xarray.DataArray object.
    This adds methods for linear algebra computations.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def to_basis(self, vector_dim):
        """
        Return a subspace basis from DataArray vectors.

        Parameters
        ----------
        vector_dim: string,
            The dimension of the vector subspace. Has to be a dimension of the DataArray.

        Returns
        -------
        basis: np.array
            A 2D numpy array with the axis representing (features, vectors)
        """
        if vector_dim not in self._obj.dims:
            raise AttributeError('{} is not a dimension of the DataArray (dims={})'.format(vector_dim, self._obj.dims))
        basis = self._obj.data.reshape(self._obj[vector_dim].size, -1).T
        return basis

    def qr_orthogonalization(self, orthogonal_dim):
        """
        Return an orthonormal subspace from DataArray vectors.

        Parameters
        ----------
        orthogonal_dim: string,
            The dimension of the vector subspace along which to orthogonalize.
            Has to be a dimension of the DataArray.


        Returns
        -------
        basis: xr.DataArray
            A DataArray with the orthogonalized subspace
        """
        q, r = scipy.linalg.qr(self._obj.linalg.to_basis(orthogonal_dim), mode='economic', overwrite_a=True)
        basis = basis_to_xarray(q, self._obj.dims, self._obj.coords, self._obj.attrs)
        return basis