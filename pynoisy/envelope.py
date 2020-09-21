"""
TODO: Some documentation and general description goes here.
"""
import noisy_core
import xarray as xr
import pynoisy.utils as utils
import numpy as np


def grid(data):
    _grid = utils.get_grid(*data.shape)
    envelope = xr.DataArray(
        name='envelope',
        data=np.array(data, dtype=np.float64, order='C'),
        coords=_grid.coords,
        dims=_grid.dims,
        attrs={'envelope_model': 'grid'}
    )
    return envelope


def ring(nx, ny, inner_radius=0.2, outer_radius=1.0, photon_ring_thickness=0.05, photon_ring_contrast=0.95,
         photon_ring_decay=100.0, ascent=1.0, inner_decay=5.0, outer_decay=10):
    """
    TODO
    """
    r = utils.get_grid(nx, ny).r.data

    zone0_radius = inner_radius
    zone1_radius = inner_radius + photon_ring_thickness

    decay1 = photon_ring_decay
    decay2 = inner_decay
    decay3 = outer_decay

    zone0 = np.exp(-1.0 / ((r + 1e-8) / (ascent * zone0_radius * 2)) ** 2)
    zone0[r > zone0_radius] = 0

    zone1 = (photon_ring_contrast + np.exp(-decay1 * (r - zone0_radius))) * np.exp(
        -1.0 / ((zone0_radius + 1e-8) / (ascent * zone0_radius * 2)) ** 2)
    zone1[r <= zone0_radius] = 0
    zone1[r > zone1_radius] = 0

    zone2 = np.exp(-decay2 * (r - zone1_radius)) * np.exp(
        -1.0 / ((zone0_radius + 1e-8) / (ascent * zone0_radius * 2)) ** 2) * \
            (photon_ring_contrast + np.exp(-decay1 * (zone1_radius - zone0_radius)))
    zone2[r <= zone1_radius] = 0

    data = zone0 + zone1 + zone2

    if outer_radius < 1.0:
        data[r > outer_radius] = 0
        zone3 = np.exp(-decay3 * (r - outer_radius)) * np.exp(-decay2 * (outer_radius - zone1_radius)) * \
                np.exp(-1.0 / ((zone0_radius + 1e-8) / (ascent * zone0_radius * 2)) ** 2) * \
                (photon_ring_contrast + np.exp(-decay1 * (zone1_radius - zone0_radius)))
        zone3[r <= outer_radius] = 0
        data += zone3

    envelope = grid(data)
    new_attrs = {
        'envelope_model': 'disk',
        'inner_radius': inner_radius,
        'outer_radius': outer_radius,
        'photon_ring_thickness': photon_ring_thickness,
        'photon_ring_contrast': photon_ring_contrast,
        'photon_ring_decay': photon_ring_decay,
        'ascent': ascent,
        'inner_decay': inner_decay,
        'outer_decay': outer_decay
    }
    envelope.attrs.update(new_attrs)
    return envelope


def noisy_ring(nx, ny, scaling_radius=0.2):
    """
    The envelope function multiplies the random fields to create synthetic images.
    The disk envelope function is a specific envelope defined by the src/model_disk.c

    Parameters
    ----------
    scaling_radius: float, default=0.02
        Scales the disk radius with respect to the whole image
    """
    envelope = grid(noisy_core.get_disk_envelope(nx, ny, scaling_radius))
    new_attrs = {
        'envelope_model': 'noisy_ring',
        'scaling_radius': scaling_radius
    }
    envelope.attrs.update(new_attrs)
    return envelope


def gaussian(std=0.2, fwhm=None):
    """TODO"""
    if fwhm is None:
        fwhm = std * np.sqrt(2 * np.log(2)) * 2 / np.sqrt(2)
    else:
        std = fwhm * np.sqrt(2) / (np.sqrt(2 * np.log(2)) * 2)

    r = utils.get_grid().r
    data = np.exp(-(r / std) ** 2)

    envelope = grid(data)
    new_attrs = {
        'envelope_model': 'gaussian',
        'std': std,
        'fwhm': fwhm
    }
    envelope.attrs.update(new_attrs)
    return envelope


def disk(nx, ny, radius=0.2, decay=20):
    """
    TODO
    """
    r = utils.get_grid(nx, ny).r
    data = np.ones_like(r)
    data[r >= .95 * radius] = 0
    exponential_decay = np.exp(-decay * (r - .95 * radius))
    data += exponential_decay.where(r >= .95 * radius).fillna(0.0)

    envelope = grid(data)
    new_attrs = {
        'envelope_model': 'disk',
        'radius': radius,
        'exponential_decay': decay
    }
    envelope.attrs.update(new_attrs)
    return envelope

def general_xy(nx, ny, scaling_radius=0.1):
    radius = utils.get_grid(nx, ny).r
    ir = scaling_radius / radius
    small = 1e-10
    envelope = (ir ** 4 * np.exp(-ir ** 2)).where(radius > scaling_radius / np.sqrt(np.log(1.0 / small))).fillna(0.0)
    new_attrs = {
        'envelope_model': 'general_xy',
        'scaling_radius': scaling_radius
    }
    envelope.attrs.update(new_attrs)
    return envelope