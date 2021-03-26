import xarray as xr
import numpy as np
import scipy.linalg
from tqdm.auto import tqdm
from scipy.sparse.linalg import lsqr


def xarray_to_basis(array):
    degree = 1
    if 'deg' in array.dims:
        degree = array.deg.size
    elif 'sample' in array.dims:
        degree = array.sample.size
    return array.data.reshape(degree, -1).T

def basis_to_xarray(basis, coords, name=None):
    if np.all([dim in coords.dims for dim in ('x', 'y')]):
        spatial_coords = ('x', 'y')
    elif np.all([dim in coords.dims for dim in ('u', 'v')]):
        spatial_coords = ('u', 'v')
    else:
        raise AttributeError('spatial coordinates are neither x,y nor u,v')


    array =  xr.DataArray(
        name=name,
        data=basis.T.reshape(-1, coords['t'].size, coords[spatial_coords[0]].size, coords[spatial_coords[1]].size),
        dims=('deg', ) + coords.dims[1:],
        coords={'deg': range(basis.shape[1]),
                't': coords['t'],
                spatial_coords[0]: coords[spatial_coords[0]],
                spatial_coords[1]: coords[spatial_coords[1]]}
    )
    return array

def project_to_subspace(vectors, subspace):
    return np.matmul(subspace, np.matmul(subspace.T, vectors))

def multiple_sources(solver, sources, n_jobs, output_type='numpy'):
    output = solver.run(source=sources, n_jobs=n_jobs, verbose=0)
    sample_size = output.sample.size if 'sample' in output.dims else 1
    output = output.data.reshape(sample_size, -1).T if output_type == 'numpy' else output
    return output

def compute_subspace(solver, input_vectors, orthogonal_subspace=None, n_jobs=4):
    basis = multiple_sources(solver, input_vectors, n_jobs)

    # Orthogonalize against subspace with double MGS (Modified Grahm Schmidt)
    if orthogonal_subspace is not None:
        basis = basis - project_to_subspace(basis, orthogonal_subspace)
        basis = basis - project_to_subspace(basis, orthogonal_subspace)

    q, r = scipy.linalg.qr(basis, mode='economic', overwrite_a=True)
    q_xr = xr.DataArray(q.T.reshape(input_vectors.shape), coords=input_vectors.coords)
    return q_xr

def randomized_subspace_iteration(solver, input_vectors, orthogonal_subspace=None, maxiter=5, n_jobs=4):
    orthogonal_subspace_degree = 0
    if orthogonal_subspace is not None:
        orthogonal_subspace_degree = orthogonal_subspace.deg.size
        orthogonal_subspace = xarray_to_basis(orthogonal_subspace)

    basis = input_vectors
    degrees = range(orthogonal_subspace_degree, orthogonal_subspace_degree + input_vectors.sample.size)
    for i in tqdm(range(maxiter), desc='subspace iteration'):
        basis = compute_subspace(solver, basis, orthogonal_subspace, n_jobs)

    basis = multiple_sources(solver, basis, n_jobs)
    u, s, v = scipy.linalg.svd(basis, full_matrices=False)

    eigenvectors = basis_to_xarray(u, input_vectors.coords, name='eigenvectors')
    eigenvalues = xr.DataArray(s, dims='deg', coords={'deg': degrees}, name='eigenvalues')
    modes = xr.merge([eigenvectors, eigenvalues])
    return modes


def real_lsqr(y, A, damp=0.0):
    l2_loss = lambda x: np.linalg.norm(y - A.dot(x))**2 + np.linalg.norm(damp*x)**2
    l2_grad = lambda x: 2*np.ascontiguousarray(np.real(A.conj().T.dot(A.dot(x) - y)) - (damp**2) * x)

    out = scipy.optimize.minimize(l2_loss, jac=l2_grad, method='L-BFGS-B',
                                  x0=np.zeros(A.shape[-1], dtype=np.float64))
    return out['fun'], out['x']

def least_squares_projection(y, A, damp=0.0, return_projection=False, return_coefs=False, real_estimation=True):
    y_array = np.array(y).ravel()
    meas_length = y.size
    y_mask = np.isfinite(y_array)
    A_mask = np.isfinite(A)

    projection = np.full_like(y_array, fill_value=np.nan)

    size_diff = None
    if meas_length < A.shape[0]:
        y_array = np.concatenate((y_array, np.zeros(A.shape[0] - meas_length)))
        y_mask = np.concatenate((y_mask, np.ones(A.shape[0] - meas_length, dtype=np.bool)))
        size_diff = meas_length - A.shape[0]

    assert np.equal(A_mask, y_mask[:, None]).all(), "Masks of A matrix and y are not identical"

    A = A[A_mask].reshape(-1, A.shape[-1])
    y_array = y_array[y_mask]

    if (real_estimation) and y.dtype == 'complex':
        r2, coefs = real_lsqr(y, A, damp)
        projection[y_mask[:meas_length]] = np.dot(A[:size_diff], coefs)
        projection = projection.reshape(*y.shape)
        r1 = np.linalg.norm(y - projection) ** 2
    else:
        out = lsqr(A, y_array, damp=damp)
        coefs, r1, r2 = out[0], out[3]**2, out[4]**2

    output = (r1, r2)
    if return_projection:
        projection[y_mask[:meas_length]] = np.dot(A[:size_diff], coefs)
        projection = projection.reshape(*y.shape)
        if isinstance(y, xr.DataArray):
            projection = xr.DataArray(projection, coords=y.coords)
        output += (projection,)
    if return_coefs:
        output += (coefs,)
    return output

def projection_residual(measurements, subspace, damp=0.0, return_projection=False, return_coefs=False):
    """
    Compute projection residual of the measurements
    """
    projection_matrix = subspace.noisy_methods.get_projection_matrix()
    result = least_squares_projection(measurements, projection_matrix, damp, return_projection, return_coefs)
    residual = xr.Dataset(data_vars={'data':result[0], 'total':result[1]}, attrs={'damp': damp})
    output = residual
    if return_projection:
        output = (residual, result[2])
    if return_coefs:
        coefs = xr.DataArray(result[-1], coords={'deg': subspace.deg}, dims='deg')
        output = (residual, coefs) if return_projection is False else output + (coefs,)
    return output