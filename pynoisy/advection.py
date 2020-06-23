"""
TODO: Some documentation and general description goes here.
"""
import noisy_core
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

def disk(nx, ny, direction='cw', scaling_radius=0.2):
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
    vx, vy = noisy_core.get_disk_velocity(nx, ny, direction_value, scaling_radius)
    advection = grid(vx, vy)
    new_attrs = {
        'advection_model': 'disk',
        'direction': direction,
        'scaling_radius': scaling_radius
    }
    advection.attrs.update(new_attrs)
    return advection