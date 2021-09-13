"""
The advection velocity fields define the motion of the stochastic features.
An advection model creates an xarray dataset with the fields: vx, vy.

References
----------
.. [1] Lee, D. and Gammie, C.F., 2021. Disks as Inhomogeneous, Anisotropic Gaussian Random Fields.
    The Astrophysical Journal, 906(1), p.39.
    url: https://iopscience.iop.org/article/10.3847/1538-4357/abc8f3/meta
"""
import numpy as _np
import xarray as _xr
import pynoisy.utils as _utils

def general_xy(ny, nx, opening_angle=-_np.pi/2, r_cutoff=0.5, grid_start=(-10, -10), grid_end=(10, 10), units='GM/c^2'):
    """
    Velocity fields (vx, vy) defined by the general_xy model [1].

    Parameters
    ----------
    ny, nx: int,
            Number of (y/x)-axis grid points.
    opening_angle: float, default=-pi/2 (clockwise rotation)
        This angle defines the opening angle with respect to the local radius.
        A negative angle rotates the velocity vector axis clockwise.
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
    grid = _utils.linspace_2d((ny, nx), grid_start, grid_end, units=units)
    magnitude = _np.abs(grid.utils_polar.w_keplerian(r_cutoff)) * grid.r
    vy = magnitude * _np.sin(grid.theta + opening_angle)
    vx = magnitude * _np.cos(grid.theta + opening_angle)

    advection = _xr.Dataset(
        data_vars={'vx': (['y','x'], vx.data),
                   'vy': (['y','x'], vy.data)},
        coords=grid.coords,
        attrs={'advection_model': 'general_xy',
               'opening_angle': opening_angle,
               'r_cutoff': r_cutoff}
    )
    return advection