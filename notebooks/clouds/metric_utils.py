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


def projection_residual(measurements, eigenvectors, degree, return_projection=False):
    """
    Compute projection residual of the measurements
    """
    vectors_reduced = eigenvectors.sel(deg=range(degree))
    coefficients = (measurements * vectors_reduced).sum(['t', 'x', 'y'])
    projection = (coefficients * vectors_reduced).sum('deg')

    residual = ((measurements - projection) ** 2).sum(['t', 'x', 'y'])
    residual.name = 'projection_residual'
    residual = residual.expand_dims(deg=[degree])

    if return_projection:
        projection.name = 'projection'
        projection = projection.expand_dims(deg=[degree])
        residual = xr.merge([residual, projection])

    return residual


def likelihood_metrics(measurements, eigenvectors, degree):
    """
    Compute likelihood metrics for eigenvector projection
    """
    vectors_reduced = eigenvectors.sel(deg=range(degree)).eigenvectors
    eigenvalues_reduced = eigenvectors.sel(deg=range(degree)).eigenvalues
    coefficients = (measurements * vectors_reduced).sum(['t', 'x', 'y'])
    projections = (coefficients * vectors_reduced).sum('deg')
    residuals = ((measurements - projections) ** 2).sum(['t', 'x', 'y'])
    quotients = ((coefficients * eigenvalues_reduced) ** 2).sum(['deg'])
    logdets = np.log(eigenvalues_reduced ** 2).sum('deg')
    likelihoods = logdets - quotients

    likelihoods.name = 'likelihood'
    quotients.name = 'quotient'
    logdets.name = 'logdet'
    residuals.name = 'projection_residual'

    dataset = xr.merge([likelihoods, quotients, logdets, residuals]).expand_dims(deg=[degree])
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