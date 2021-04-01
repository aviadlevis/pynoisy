"""
TODO: Some documentation and general description goes here.
"""
import numpy as np
import xarray as xr
import pynoisy.utils as utils

def ring(nx, ny, fov=1.0, inner_radius=0.17, outer_radius=1.0, photon_ring_thickness=0.05, photon_ring_contrast=0.95,
         photon_ring_decay=100.0, ascent=1.0, inner_decay=8.0, outer_decay=10, total_flux=2.0):
    """
    Ring envelope with a brighter photon ring in the inner-most orbit.

    Parameters
    ----------
    nx: int,
        Number of x-axis grid points.
    ny: int,
        Number of y-axis grid points.
    fov: float, default=1.0,
        Field of view. Default is unitless 1.0.
    inner_radius: float, default=0.2,
        inner radius of the black-hole shadow.
    outer_radius: float, default=1.0,
        Cutoff outer radius for the exponential decay of flux. Beyond this radius the flux it cutoff to zero.
    photon_ring_thickness: float, default=0.05,
        Thickness of the inner bright photon ring.

    """
    grid = utils.linspace_2d((nx, ny), (-fov/2.0, -fov/2.0), (fov/2.0, fov/2.0))
    r = grid.r.data

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

    envelope = xr.DataArray(
        name='envelope',
        data=np.array(data, dtype=np.float64, order='C'),
        coords=grid.coords,
        attrs={
            'envelope_model': 'ring',
            'fov': fov,
            'inner_radius': inner_radius,
            'outer_radius': outer_radius,
            'photon_ring_thickness': photon_ring_thickness,
            'photon_ring_contrast': photon_ring_contrast,
            'photon_ring_decay': photon_ring_decay,
            'ascent': ascent,
            'inner_decay': inner_decay,
            'outer_decay': outer_decay,
            'total_flux': total_flux
        } )

    envelope *= (total_flux / envelope.sum())
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


def azimuth_symetric(coords, envelope_r):
    """
    TODO
    """
    envelope = envelope_r.interp(r=coords['r'])
    envelope.attrs.update(envelope_model='azimuth_symetric')
    return envelope