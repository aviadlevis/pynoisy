"""
TODO: Some documentation and general description goes here.
"""
import noisy_core, hgrf_core
import numpy as np
import xarray as xr
import pynoisy.utils as utils

def grid(spatial_angle, correlation_time, correlation_length, tensor_ratio, diffusion_coefficient=None):
    assert spatial_angle.shape == correlation_time.shape  == correlation_length.shape, \
        'shapes of (spatial_angle, correlation_time, correlation_length) have to match.'
    _grid = utils.get_grid(*spatial_angle.shape)
    if diffusion_coefficient is None:
        diffusion_coefficient = 2.0 * correlation_length ** 2 / correlation_time
    else:
        correlation_length = np.sqrt(0.5 * diffusion_coefficient * correlation_time)
    diffusion = xr.Dataset(
        data_vars={
            'spatial_angle': (_grid.dims, spatial_angle),
            'correlation_time': (_grid.dims, correlation_time),
            'correlation_length': (_grid.dims, correlation_length),
            'diffusion_coefficient': (_grid.dims, diffusion_coefficient),
            'tensor_ratio': tensor_ratio
        },
        coords=_grid.coords,
        attrs={'diffusion_model': 'grid'}
    )
    return diffusion

def ring(nx, ny, tensor_ratio=0.1, opening_angle=np.pi / 3.0, tau=1.0, lam=0.5, scaling_radius=0.2):
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
        spatial_angle=noisy_core.get_disk_angle(nx, ny, opening_angle),
        correlation_time=noisy_core.get_disk_correlation_time(nx, ny, tau, scaling_radius),
        correlation_length=noisy_core.get_disk_correlation_length(nx, ny, scaling_radius, lam),
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

def multivariate_gaussian(nx, ny, length_scale=0.1, tensor_ratio=0.1, max_diffusion_coef=1.0, max_correlation_time=1.0):
    """
    TODO

    Parameters
    ----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    tensor_ratio: float, default=0.1
        ratio for the diffusion tensor along the two principal axis.
    """
    covariance = utils.matern_covariance(length_scale=length_scale)

    print('Sampling diffusion principle angle from a multivariate gaussian (Matern covariance kernel size = {}x{})'.format(*covariance.shape))
    spatial_angle = np.random.multivariate_normal(np.zeros(x.size), covariance)

    print('Sampling correlation time from a multivariate gaussian (Matern covariance kernel size = {}x{})'.format(*covariance.shape))
    correlation_time = np.random.multivariate_normal(np.zeros(x.size), covariance)
    correlation_time = np.exp(correlation_time)
    correlation_time = max_correlation_time * correlation_time / correlation_time.max()

    print('Sampling diffusion coefficient from a multivariate gaussian (Matern covariance kernel size = {}x{})'.format(*covariance.shape))
    diffusion_coefficient = np.random.multivariate_normal(np.zeros(x.size), covariance)
    diffusion_coefficient = max_diffusion_coef * (diffusion_coefficient - diffusion_coefficient.min()) / \
                            (diffusion_coefficient.max() - diffusion_coefficient.min())

    # diffusion_coefficient = 2.0 * correlation_length ** 2 / correlation_time
    correlation_length = np.sqrt(0.5 * diffusion_coefficient * correlation_time)

    diffusion = grid(
        spatial_angle=spatial_angle.reshape(x.shape),
        correlation_time=correlation_time.reshape(x.shape),
        correlation_length=correlation_length.reshape(x.shape),
        tensor_ratio=tensor_ratio
    )
    diffusion['covariance'] = xr.DataArray(covariance, coords=[x.ravel(), y.ravel()], dims=['i', 'j'])
    new_attrs = {
        'diffusion_model': 'multivariate_gaussian',
        'kernel': 'Matern',
        'length_scale': length_scale,
        'max_diffusion_coef': max_diffusion_coef,
        'max_corr_time': max_correlation_time,
        'spatial_angle_model': 'unnormalized',
        'corr_time_model': 'normalized (exponential)',
        'diffusion_coef_model': 'normalized (linear)'
    }
    diffusion.attrs.update(new_attrs)
    return diffusion

def disk(nx, ny, tensor_ratio=0.1, direction='cw', tau=1.0, scaling_radius=0.2):
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
        spatial_angle=noisy_core.get_disk_angle(nx, ny, -direction * np.pi / 2),
        correlation_time=noisy_core.get_disk_correlation_time(nx, ny, tau, scaling_radius),
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

def general_xy(nx, ny, opening_angle=np.pi/2 - np.pi/9, tau=1.0, lam=5.0, tensor_ratio=0.1, scaling_radius=0.5):
    """
    TODO

    Parameters
    ----------
    opening_angle: float, default= np.pi/2 - np.pi/9
        This angle defines the opening angle of spirals with respect to the local radius
    tau: float, default=1.0
        product of correlation time and local Keplerian frequency.
    lam: float, default=5.0
        ratio of correlation length to local radius
    tensor_ratio (r12): float, default=0.1
        ratio for the diffusion tensor along the two principal axis.
    scaling_radius: float, default=0.5
        scaling parameter for the Kepler orbital frequency (the magnitude of the velocity).
    """
    diffusion = grid(
        spatial_angle=hgrf_core.get_generalxy_spatial_angle(nx, ny, scaling_radius, opening_angle),
        correlation_time=hgrf_core.get_generalxy_correlation_time(nx, ny, tau, scaling_radius),
        correlation_length=hgrf_core.get_generalxy_correlation_length(nx, ny, scaling_radius, lam),
        tensor_ratio=tensor_ratio
    )
    new_attrs = {
            'diffusion_model': 'general_xy',
            'opening_angle': opening_angle,
            'tau': tau,
            'lam': lam,
            'scaling_radius': scaling_radius
    }
    diffusion.attrs.update(new_attrs)
    return diffusion




