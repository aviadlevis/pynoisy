"""
The diffusion fields define the spatio-temporal correlations of the stochastic features.
A diffusion model creates an xarray dataset with the fields:
    correlation_length, correlation_time, spatial_angle, tensor_radio (scalar)

References
----------
.. [1] Lee, D. and Gammie, C.F., 2021. Disks as Inhomogeneous, Anisotropic Gaussian Random Fields.
    The Astrophysical Journal, 906(1), p.39.
    url: https://iopscience.iop.org/article/10.3847/1538-4357/abc8f3/meta
"""
import numpy as np
import xarray as xr
import pynoisy.utils

def general_xy(ny, nx, opening_angle=np.pi/2 - np.pi/9, tau=1.0, lam=5.0, tensor_ratio=0.1, r_cutoff=0.5,
               grid_start=(-10, -10), grid_end=(10, 10), units='GM/c^2'):
    """
    Diffusion fields defined by the general_xy model [1].

    Parameters
    ----------
    ny, nx: int,
            Number of (y/x)-axis grid points.
    opening_angle: float, default = np.pi/2 - np.pi/9
        This angle defines the opening angle of spirals with respect to the local radius.
        A positive angle rotates the correlation to match clockwise rotation trailing spiral arms.
        (this means counter clockwise rotation of the local correlation axes)
    tau: float, default=1.0
        Product of correlation time and local Keplerian frequency.
    lam: float, default=5.0
        Ratio of correlation length to local radius
    tensor_ratio (r12): float, default=0.1
        Ratio for the diffusion tensor along the two principal axis.
    r_cutoff: float
        Cutoff radius for a smooth center point.
    grid_start: (float, float)
        (x, y) starting grid point (included in the grid).
    grid_end: (float, float), default=(-10, 10)
        (x, y) ending grid point (end point not included in the grid).
    units: str, default='GM/c^2',
        Units of x/y in terms of M.

    Returns
    -------
    diffusion: xr.Dataset
        A Dataset containing the following fields:
            - correlation_length (2D field)
            - correlation_time (2D field)
            - spatial_angle (2D field)
            - tensor_radio (scalar)

    References
    ----------
    .. [1] inoisy code: https://github.com/AFD-Illinois/inoisy
    """
    grid = pynoisy.utils.linspace_2d((ny, nx), grid_start, grid_end, units=units)
    correlation_time = general_xy_correlation_time(grid.r, tau, r_cutoff)
    correlation_length = general_xy_correlation_length(grid.r, lam, r_cutoff)
    spatial_angle = general_xy_spatial_angle(grid.theta, opening_angle)

    diffusion = xr.Dataset(
        data_vars={
            'spatial_angle': (['y', 'x'], spatial_angle),
            'correlation_time': (['y', 'x'], correlation_time),
            'correlation_length': (['y', 'x'], correlation_length),
            'tensor_ratio': tensor_ratio
        },
        coords=grid.coords,
        attrs={'diffusion_model': 'general_xy'}
    )
    diffusion.attrs.update(correlation_time.attrs)
    diffusion.attrs.update(correlation_length.attrs)
    diffusion.attrs.update(spatial_angle.attrs)
    return diffusion

def general_xy_correlation_length(r, lam=5.0, r_cutoff=0.5):
    """
    Compute azimuthal symetric correlation length on a grid according to general_xy.
    Source: inoisy/src/param_general_xy.c

    Parameters
    ----------
    r: xr.DataArray
        A DataArray with the radial coordinates on a 2D grid.
    lam: float, default=5.0
        ratio of correlation length to local radius
    r_cutoff: float
        Cutoff radius for a smooth center point.

    Returns
    -------
    correlation_time: xr.DataArray
        A DataArray with corrlation length on a grid.

    References
    ----------
    https://github.com/aviadlevis/inoisy/blob/47fb41402ecdf93bfdd176fec780e8f0ba43445d/src/param_general_xy.c#L156
    """
    correlation_length = lam * r
    if r_cutoff > 0.0:
        correlation_length.values[(r < r_cutoff).data] = correlation_length.polar.r_cutoff(
            r_cutoff, lam * r_cutoff, lam, 0.9 * lam * r_cutoff).values[(r < r_cutoff).data]
    correlation_length.name = 'correlation_length'
    correlation_length.attrs.update({
        'lam': lam,
        'r_cutoff': r_cutoff
    })
    return correlation_length

def general_xy_correlation_time(r, tau=1.0, r_cutoff=0.5):
    """
    Compute azimuthal symetric correlation time on a grid according to general_xy.
    Source: inoisy/src/param_general_xy.c

    Parameters
    ----------
    r: xr.DataArray
        A DataArray with the radial coordinates on a 2D grid.
    tau: float, default=1.0
        product of correlation time and local Keplerian frequency.
    r_cutoff: float
        Cutoff radius for a smooth center point.

    Returns
    -------
    correlation_time: xr.DataArray
        A DataArray with corrlation time on a grid.

    References
    ----------
    https://github.com/aviadlevis/inoisy/blob/47fb41402ecdf93bfdd176fec780e8f0ba43445d/src/param_general_xy.c#L169
    """
    correlation_time = 2.0 * np.pi * tau / np.abs(r.polar.w_keplerian(r_cutoff))
    if r_cutoff > 0.0:
        correlation_time.values[(r < r_cutoff).data] = correlation_time.polar.r_cutoff(
            r_cutoff, 2 * np.pi * tau * r_cutoff ** 1.5, 2 * np.pi * tau * 1.5 * np.sqrt(r_cutoff),
                      0.9 * 2 * np.pi * tau * r_cutoff ** 1.5).values[(r < r_cutoff).data]
    correlation_time.name = 'correlation_time'
    correlation_time.attrs.update({
        'tau': tau,
        'r_cutoff': r_cutoff
    })
    return correlation_time

def general_xy_spatial_angle(theta, opening_angle=np.pi/2 - np.pi/9):
    """
    Compute angle of spatial correlation on a grid according to general_xy.
    Source: inoisy/src/param_general_xy.c

    Parameters
    ----------
    theta: xr.DataArray
        A DataArray with the azimuthal coordinates on a 2D grid.
    opening_angle: float, default = np.pi/2 - np.pi/9
        This angle defines the opening angle of spirals with respect to the local radius.
        A positive angle rotates the correlation to match clockwise rotation trailing spiral arms.
        (this means counter clockwise rotation of the local correlation axes)
    Returns
    -------
    spatial_angle: xr.DataArray
        A DataArray with angle of spatial correlation on a grid.

    References
    ----------
    https://github.com/aviadlevis/inoisy/blob/47fb41402ecdf93bfdd176fec780e8f0ba43445d/src/param_general_xy.c#L213
    """
    spatial_angle = theta + opening_angle
    spatial_angle.name = 'spatial_angle'
    spatial_angle.attrs.update(opening_angle=opening_angle)
    return spatial_angle




