"""
Envelopes are static images which capture stationary features of the movie (e.g. black hole shadow).
Simple geometric image structures are used to model the static envelope (e.g. ring, gaussian, disk etc..)
"""
import numpy as _np
import xarray as _xr
import pynoisy.utils as _utils

def ring(ny, nx, fov=(1.0, 'unitless'), inner_radius=0.17, outer_radius=1.0, photon_ring_thickness=0.05, photon_ring_contrast=0.95,
         photon_ring_decay=100.0, ascent=1.0, inner_decay=8.0, outer_decay=10, total_flux=1.0):
    """
    Ring envelope with a brighter photon ring in the inner-most orbit [1].

    Parameters
    ----------
    ny, nx: int,
            Number of (y/x)-axis grid points.
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    inner_radius: float, default=0.17,
        inner radius of the black-hole shadow.
    outer_radius: float, default=1.0,
        Cutoff outer radius for the exponential decay of flux. Beyond this radius the flux it cutoff to zero.
    photon_ring_thickness: float, default=0.05,
        Thickness of the inner bright photon ring.
    photon_ring_contrast: float, default=0.95,
        Brightness of the inner photon ring.
    photon_ring_decay: float, default=100.0,
        Exponential decay rate of the photon ring
    ascent: float, 1.0,
        Decay rate of the black-hole shadow region.
    inner_decay: float, default=8.0,
        Exponential decay rate of the inner ring.
    outer_decay: float, default=10.0,
        Exponential decay rate of the outer ring.
    total_flux: float, default=1.0,
        Total flux normalization of the image (sum over pixels).

    Returns
    -------
    envelope: xr.DataArray,
        An image DataArray with dimensions ['y', 'x'].

    References
    ----------
    .. [1] Narayan, R., Johnson, M.D. and Gammie, C.F., The shadow of a spherically accreting black hole.
           The Astrophysical Journal Letters, 885(2), p.L33. 2019.
           url: https://iopscience.iop.org/article/10.3847/2041-8213/ab518c/pdf
    """
    grid = _utils.linspace_2d((ny, nx), (-fov[0]/2.0, -fov[0]/2.0), (fov[0]/2.0, fov[0]/2.0), units=fov[1])
    r = grid.r.data

    zone0_radius = inner_radius
    zone1_radius = inner_radius + photon_ring_thickness

    decay1 = photon_ring_decay
    decay2 = inner_decay
    decay3 = outer_decay

    zone0 = _np.exp(-1.0 / ((r + 1e-8) / (ascent * zone0_radius * 2)) ** 2)
    zone0[r > zone0_radius] = 0

    zone1 = (photon_ring_contrast + _np.exp(-decay1 * (r - zone0_radius))) * _np.exp(
        -1.0 / ((zone0_radius + 1e-8) / (ascent * zone0_radius * 2)) ** 2)
    zone1[r <= zone0_radius] = 0
    zone1[r > zone1_radius] = 0

    zone2 = _np.exp(-decay2 * (r - zone1_radius)) * _np.exp(
        -1.0 / ((zone0_radius + 1e-8) / (ascent * zone0_radius * 2)) ** 2) * \
            (photon_ring_contrast + _np.exp(-decay1 * (zone1_radius - zone0_radius)))
    zone2[r <= zone1_radius] = 0

    data = zone0 + zone1 + zone2

    if outer_radius < 1.0:
        data[r > outer_radius] = 0
        zone3 = _np.exp(-decay3 * (r - outer_radius)) * _np.exp(-decay2 * (outer_radius - zone1_radius)) * \
                _np.exp(-1.0 / ((zone0_radius + 1e-8) / (ascent * zone0_radius * 2)) ** 2) * \
                (photon_ring_contrast + _np.exp(-decay1 * (zone1_radius - zone0_radius)))
        zone3[r <= outer_radius] = 0
        data += zone3

    envelope = _xr.DataArray(
        name='envelope',
        data=_np.array(data, dtype=_np.float64, order='C'),
        coords=grid.coords,
        dims=['y', 'x'],
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
        })

    envelope *= (total_flux / envelope.sum())
    return envelope

def gaussian(ny, nx, fov=(1.0, 'unitless'), std=0.2, fwhm=None, total_flux=1.0):
    """
    Gaussian envelope.

    Parameters
    ----------
    ny, nx: int,
            Number of (y/x)-axis grid points.
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    std: float, default=0.2, optional,
        Gaussian standard deviation. Used if fwhm is not specified.
    fwhm: float, optional,
        Gaussian full width half max. Overrides the std parameter.
    total_flux: float, default=1.0,
        Total flux normalization of the image (sum over pixels).

    Returns
    -------
    envelope: xr.DataArray,
        An image DataArray with dimensions ['y', 'x'].
    """
    if fwhm is None:
        fwhm = 2 * _np.sqrt(2 * _np.log(2)) * std
    else:
        std = fwhm / (2 * _np.sqrt(2 * _np.log(2)))

    grid = _utils.linspace_2d((ny, nx), (-fov[0] / 2.0, -fov[0] / 2.0), (fov[0] / 2.0, fov[0] / 2.0), units=fov[1])
    r = grid.r.data
    data = _np.exp(-0.5*(r / std) ** 2)

    envelope = _xr.DataArray(
        name='envelope',
        data=_np.array(data, dtype=_np.float64, order='C'),
        coords=grid.coords,
        dims=['y', 'x'],
        attrs={
            'envelope_model': 'gaussian',
            'fov': fov,
            'std': std,
            'fwhm': fwhm,
            'total_flux': total_flux
        })
    envelope *= (total_flux / envelope.sum())
    return envelope

def disk(ny, nx, fov=(1.0, 'unitless'), radius=0.2, decay=20, total_flux=1.0):
    """
    Disk envelope.

    Parameters
    ----------
    ny, nx: int,
            Number of (y/x)-axis grid points.
    fov: (float, str), default=(1.0, 'unitless')
        Field of view and units. Default is unitless 1.0.
    radius: float, default=0.2, optional,
        Disk radius.
    decay: float, default=20.
        Exponential decay rate of flux close to the disk edge (at 95% of the disk radius).
    total_flux: float, default=1.0,
        Total flux normalization of the image (sum over pixels).

    Returns
    -------
    envelope: xr.DataArray,
        An image DataArray with dimensions ['y', 'x'].
    """
    grid = _utils.linspace_2d((ny, nx), (-fov[0] / 2.0, -fov[0] / 2.0), (fov[0] / 2.0, fov[0] / 2.0), units=fov[1])
    data = _utils.full_like(grid.coords, fill_value=1.0)
    data.values[data.r >= .95 * radius] = 0
    exponential_decay = _np.exp(-decay * (data.r - .95 * radius))
    data += exponential_decay.where(data.r >= .95 * radius).fillna(0.0)

    envelope = _xr.DataArray(
        name='envelope',
        data=_np.array(data, dtype=_np.float64, order='C'),
        coords=grid.coords,
        dims=['y', 'x'],
        attrs={
            'envelope_model': 'disk',
            'fov': fov,
            'radius': radius,
            'decay': decay,
            'total_flux': total_flux
        })
    envelope *= (total_flux / envelope.sum())
    return envelope