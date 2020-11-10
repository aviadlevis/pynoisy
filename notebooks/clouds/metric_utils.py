import pynoisy
import numpy as np
import xarray as xr
from tqdm.notebook import tqdm

def compute_full_quotient(measurements, hyperparams, param_grid):
    """
    Compute the full quotient for parameter
    :param measurements: Measurements (video)
    :type measurements: xr.DataArray(nt,nx,ny)
    :param hyperparams: {'nt': int,
                        'nx': int,
                        'ny': int,
                        'wind_angle':float,
                        'wind_magnitude': float,
                        'correlation_time': float,
                        'correlation_length': float,
                        'evolution_length': float,
                        'spatial_angle': float,
                        'tensor_ratio': float}
    :type hyperparams: dict
    :param param_grid: {'param': range of values}
    :type param_grid: dict
    :return: full_quotient
    :rtype: list
    """
    full_quotient = []
    param_name, param_values = list(param_grid.items())[0]
    for param_value in tqdm(param_values, leave=False):
        hyperparams.update({param_name: param_value})
        solver = pynoisy.forward.HGRFSolver.homogeneous(**hyperparams)
        y = -solver.get_laplacian(measurements)
        quotient = (y * y).sum().data
        full_quotient.append(quotient)
    return full_quotient


def compute_krylov_loss(measurements, hyperparams, param_grid, degrees):
    krylov_residuals = []
    param_name, param_values = list(param_grid.items())[0]
    for degree in tqdm(degrees, desc='degree'):
        krylov = []
        for param_value in tqdm(param_values, desc='angle', leave=False):
            hyperparams.update({param_name: param_value})
            solver = pynoisy.forward.HGRFSolver.homogeneous(**hyperparams)
            krylov.append(pynoisy.utils.krylov_residual(solver, measurements, degree))

        krylov = xr.DataArray(krylov, dims=param_name,
                              coords={param_name: param_values},
                              name='krylov_residual').expand_dims(degree=[degree])
        krylov_residuals.append(krylov)
    output = xr.merge(krylov_residuals)
    return output


def likelihood_metrics(measurements, eigenvectors, degree):
    """
    Compute likelihood metrics for eigenvector projection
    """
    likelihoods, quotients, logdets, projection_residuals = [], [], [], []
    metric_dim = eigenvectors[0].dims[0]
    for vectors in eigenvectors:
        vectors_reduced = vectors.sel(deg=range(degree))
        coefficients = (measurements * vectors_reduced).sum(['t','x','y'])
        quotient = ((coefficients * vectors_reduced.eigenvalue)**2).sum(['deg'])
        logdet = np.log(vectors_reduced.eigenvalue**2).sum().expand_dims(
            {metric_dim: vectors_reduced[metric_dim]}
        )
        residual = ((measurements - coefficients*vectors_reduced)**2).sum(['deg','t','x','y'])
        quotients.append(quotient)
        logdets.append(logdet)
        likelihoods.append(logdet + quotient)
        projection_residuals.append(residual)

    likelihoods = xr.concat(likelihoods, dim=metric_dim)
    logdets = xr.concat(logdets, dim=metric_dim)
    quotients = xr.concat(quotients, dim=metric_dim)
    projection_residuals = xr.concat(projection_residuals, dim=metric_dim)
    likelihoods.name = 'likelihood'
    quotients.name = 'quotient'
    logdets.name = 'logdet'
    projection_residuals.name = 'projection_residual'
    dataset = xr.merge([likelihoods, quotients, logdets, projection_residuals]).expand_dims({'degree': [degree]})
    return dataset


def objective_fun(solver, measurements, degree, n_jobs=4):
    error = krylov_error_fn(solver, measurements, degree, n_jobs)
    loss = (error ** 2).mean()
    return np.array(loss)


def krylov_error_fn(solver, measurements, degree, n_jobs=4):
    krylov = solver.run(source=measurements, nrecur=degree, verbose=0, std_scaling=False, n_jobs=n_jobs)
    k_matrix = krylov.data.reshape(degree, -1)
    result = np.linalg.lstsq(k_matrix.T, np.array(measurements).ravel(), rcond=-1)
    coefs, residual = result[0], result[1]
    random_field = np.dot(coefs.T, k_matrix).reshape(*measurements.shape)
    error = random_field - measurements
    return error