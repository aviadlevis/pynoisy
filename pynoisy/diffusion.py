"""
TODO: Some documentation and general description goes here.
"""
import core
import numpy as np
import xarray as xr
import pynoisy.utils as utils

def grid(principle_angle, correlation_time, correlation_length, tensor_ratio):
    _grid = utils.get_grid()
    diffusion_coefficient = 2.0 * correlation_length ** 2 / correlation_time
    diffusion = xr.Dataset(
        data_vars={
            'principle_angle': (_grid.dims, principle_angle),
            'correlation_time': (_grid.dims, correlation_time),
            'correlation_length': (_grid.dims, correlation_length),
            'diffusion_coefficient': (_grid.dims, diffusion_coefficient),
            'tensor_ratio': tensor_ratio
        },
        coords=_grid.coords,
        attrs={'diffusion_model': 'grid'}
    )
    return diffusion

def ring(tensor_ratio=0.1, opening_angle=np.pi / 3.0, tau=1.0, lam=0.5, scaling_radius=0.2):
    """
    TODO

    Parameters
    ----------
    opening_angle: float, default= pi/3
        This angle defines the opening angle of spirals with respect to the local radius
    tau: float, default=1.0
        product of correlation time and local Keplerian frequency
    lam: float, default=0.5
        ratio of correlation length to local radius
    scaling_radius: float, default=0.2
        scaling parameter for the Kepler orbital frequency (the magnitude of the velocity)
    tensor_ratio: float, default=0.1
        ratio for the diffusion tensor along the two principal axis.
    """
    diffusion = grid(
        principle_angle=core.get_disk_angle(opening_angle),
        correlation_time=core.get_disk_correlation_time(tau, scaling_radius),
        correlation_length=core.get_disk_correlation_length(scaling_radius, lam),
        tensor_ratio=tensor_ratio
    )
    new_attrs = {
        'diffusion_model': 'ring',
        'opening_angle': opening_angle,
        'tau': tau,
        'lam': lam,
        'scaling_radius': scaling_radius
    }
    diffusion.attrs.update(new_attrs)
    return diffusion

def multivariate_gaussian(length_scale=0.1, tensor_ratio=0.1, tau=1.0, lam=0.5, scaling_radius=0.2):
    """
    TODO

    Parameters
    ----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    tau: float, default=1.0
        product of correlation time and local Keplerian frequency
    lam: float, default=0.5
        ratio of correlation length to local radius
    scaling_radius: float, default=0.2
        scaling parameter for the Kepler orbital frequency (the magnitude of the velocity)
    tensor_ratio: float, default=0.1
        ratio for the diffusion tensor along the two principal axis.
    """
    from sklearn.gaussian_process.kernels import Matern
    kernel = Matern(length_scale=length_scale)
    _grid = utils.get_grid()
    x, y = np.meshgrid(_grid.x, _grid.y)
    covariance = kernel(np.array([x.ravel(), y.ravel()]).T)
    print('Sampling diffusion principle angle from a multivariate gaussian (Matern covariance kernel size = {}x{})'.format(*covariance.shape))
    principle_angle = np.random.multivariate_normal(np.zeros(x.size), covariance)
    diffusion = grid(
        principle_angle=principle_angle.reshape(x.shape),
        correlation_time=core.get_disk_correlation_time(tau, scaling_radius),
        correlation_length=core.get_disk_correlation_length(scaling_radius, lam),
        tensor_ratio=tensor_ratio
    )
    diffusion['covariance'] = xr.DataArray(covariance, coords=[x.ravel(), y.ravel()], dims=['i', 'j'])
    new_attrs = {
        'diffusion_model': 'multivariate_gaussian',
        'kernel': 'Matern',
        'length_scale': length_scale,
        'tau': tau,
        'lam': lam,
        'scaling_radius': scaling_radius
    }
    diffusion.attrs.update(new_attrs)
    return diffusion

def disk(tensor_ratio=0.1, direction='cw', tau=1.0, scaling_radius=0.2):
    """
    TODO

    Parameters
    ----------
    direction: str, default='cw'
        'cw' or 'ccw' for clockwise or counter-clockwise directions.
    tau: float, default=1.0
        product of correlation time and local Keplerian frequency.
    scaling_radius: float, default=0.2
        scaling parameter for the Kepler orbital frequency (the magnitude of the velocity).
    tensor_ratio: float, default=0.1
        ratio for the diffusion tensor along the two principal axis.
    """
    assert direction in ['cw', 'ccw'], 'Direction can be either cw or ccw, not {}'.format(direction)
    direction = -1 if direction is 'cw' else 1

    diffusion = grid(
        principle_angle=core.get_disk_angle(-direction * np.pi / 2),
        correlation_time=core.get_disk_correlation_time(tau, scaling_radius),
        correlation_length=np.exp(-0.5 * (utils.get_grid().r / scaling_radius) ** 2),
        tensor_ratio=tensor_ratio
    )
    new_attrs = {
        'diffusion_model': 'disk',
        'tensor_ratio': tensor_ratio,
        'direction': direction,
        'tau': tau,
        'scaling_radius': scaling_radius
    }
    diffusion.attrs.update(new_attrs)
    return diffusion