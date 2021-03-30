"""
The advection velocity fields define the motion of the stochastic features.
An advection model creates an xarray dataset with the fields: vx, vy.

References
----------
.. [1] Lee, D. and Gammie, C.F., 2021. Disks as Inhomogeneous, Anisotropic Gaussian Random Fields.
    The Astrophysical Journal, 906(1), p.39.
    url: https://iopscience.iop.org/article/10.3847/1538-4357/abc8f3/meta
"""
import numpy as np
import xarray as xr
import pynoisy.utils

def general_xy(nx, ny, opening_angle=0.0, direction='ccw', r_cutoff=0.5, grid_start=(-10, -10), grid_end=(10, 10)):
    """
    Velocity fields (vx, vy) defined by the general_xy model [1].

    Parameters
    ----------
    nx: int,
        Number of x-axis grid points.
    ny: int,
        Number of y-axis grid points.
    opening_angle: float, default=0.0
        This angle defines the opening angle respect to the local azimuthal angle.
        opening angle=0.0 is strictly rotational movement.
    direction: str, default='ccw'
        'cw' or 'ccw' for clockwise or counter-clockwise directions.
    r_cutoff: float
        Cutoff radius for a smooth center point.

    Returns
    -------
    advection: xr.Dataset
        A Dataset containing the velocity field: (vx, vy) on a grid.

    References
    ----------
    .. [1] inoisy code: https://github.com/AFD-Illinois/inoisy
    """
    if direction == 'ccw':
        direction_value = -1
    elif direction == 'cw':
        direction_value = 1
    else:
        raise AttributeError('Direction can be either cw or ccw, not {}'.format(direction))

    grid = pynoisy.utils.linspace_2d((nx, ny), grid_start, grid_end)
    magnitude = np.abs(grid.polar.w_keplerian(r_cutoff)) * grid.r
    vx = direction_value * magnitude * np.cos(-grid.theta - opening_angle)
    vy = direction_value * magnitude * np.sin(-grid.theta - opening_angle)

    advection = xr.Dataset(
        data_vars={'vx': (grid.dims, vx), 'vy': (grid.dims, vy)},
        coords=grid.coords,
        attrs={'advection_model': 'general_xy',
               'opening_angle': opening_angle,
               'direction': direction,
               'r_cutoff': r_cutoff}
    )
    return advection