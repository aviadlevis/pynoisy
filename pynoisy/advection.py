"""
TODO: Some documentation and general description goes here.
"""
import noisy_core, hgrf_core
import xarray as xr
import pynoisy.utils as utils

def grid(vx, vy):
    assert vx.shape == vy.shape, 'vx.shape doesnt match vy.shape'
    _grid = utils.get_grid(*vx.shape)
    advection = xr.Dataset(
        data_vars={'vx': (_grid.dims, vx), 'vy': (_grid.dims, vy)},
        coords=_grid.coords,
        attrs={'advection_model': 'grid'}
    )
    return advection

def disk(nx, ny, direction='ccw', scaling_radius=0.2):
    """
    TODO

    Parameters
    ----------
    direction: str, default='cw'
        'cw' or 'ccw' for clockwise or counter-clockwise directions.
    scaling_radius: float, default=0.2
        scaling parameter for the Kepler orbital frequency (the magnitude of the velocity)
    """
    assert direction in ['cw', 'ccw'], 'Direction can be either cw or ccw, not {}'.format(direction)
    direction_value = -1 if direction is 'cw' else 1
    vy, vx = noisy_core.get_disk_velocity(nx, ny, direction_value, scaling_radius)
    advection = grid(vx, vy)
    new_attrs = {
        'advection_model': 'disk',
        'direction': direction,
        'scaling_radius': scaling_radius
    }
    advection.attrs.update(new_attrs)
    return advection

def general_xy(nx, ny, direction='ccw', scaling_radius=0.5, opening_angle=0.0):
    """
    TODO

    Parameters
    ----------
    direction: str, default='ccw'
        'cw' or 'ccw' for clockwise or counter-clockwise directions.
    scaling_radius: float, default=0.5
        scaling parameter for the Kepler orbital frequency (the magnitude of the velocity)
    opening_angle: float, default=0.0
        This angle defines the opening angle respect to the local azimuthal angle.
        opening angle=0.0 is strictly rotational movement
    """
    assert direction in ['cw', 'ccw'], 'Direction can be either cw or ccw, not {}'.format(direction)
    direction_value = -1 if direction is 'ccw' else 1
    vy, vx = hgrf_core.get_generalxy_velocity(nx, ny, direction_value, scaling_radius, opening_angle)
    advection = grid(vx, vy)
    new_attrs = {
        'advection_model': 'general_xy',
        'direction': direction,
        'scaling_radius': scaling_radius,
        'opening_angle': opening_angle
    }
    advection.attrs.update(new_attrs)
    return advection