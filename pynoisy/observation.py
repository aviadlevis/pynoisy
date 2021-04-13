"""
Observations classes and functions to compute EHT interferometric observations.
Here is where the interface with eht-imaging [1] library resides.

References
----------
.. [1] eht-imaging: https://github.com/achael/eht-imaging
"""
import xarray as xr
import numpy as np
import ehtim.observing.obs_helpers as obsh
import ehtim as eh
import ehtim.const_def as ehc


def block_observe_same_nonoise(movies, psize, obs, fft_pad_factor=2, conjugate=False):
    """
    Modification of ehtim's observe_same_nonoise method for a block of movies.

    Parameters
    ----------
    movies: xr.DataArray,
        a 3D/4D dataarray with a single movie or block of multiple movies.
        dimensions are [...,'t','y','x']
    psize: float,
        pixel size in [rad]
    obs (Obsdata): ehtim observation data containing the array geometry and measurement times.
                  If obs=None this function returns the full Fourier transform.
    fft_pad_factor (float):  a padding factor for increased fft resolution.
    conjugate (bool): negative frequencies (conjugate for real data)
    Returns:
        block_vis(xr.DataArray): a data array of shape (num_movies, num_visibilities) and dtype np.complex128.

    Notes:
        1. NFFT is not supported.
        2. Cubic interpolation in both time and uv (ehtim has linear interpolation in time and cubic in uv).
    Refs:
        https://github.com/achael/eht-imaging/blob/6b87cdc65bdefa4d9c4530ea6b69df9adc531c0c/ehtim/movie.py#L981
        https://github.com/achael/eht-imaging/blob/6b87cdc65bdefa4d9c4530ea6b69df9adc531c0c/ehtim/observing/obs_simulate.py#L182
        https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    """
    movies = movies.squeeze()

    # Pad images according to pad factor (interpolation in Fourier space)
    npad = fft_pad_factor * np.max((movies.x.size, movies.y.size))
    npad = obsh.power_of_two(npad)
    padvalx1 = padvalx2 = int(np.floor((npad - movies.x.size) / 2.0))
    if movies.x.size % 2:
        padvalx2 += 1
    padvaly1 = padvaly2 = int(np.floor((npad - movies.y.size) / 2.0))
    if movies.y.size % 2:
        padvaly2 += 1
    padded_movies = movies.pad({'x': (padvalx1, padvalx2), 'y': (padvaly1, padvaly2)}, constant_values=0.0)

    # Compute visibilities (Fourier transform) of the entire block
    freqs = np.fft.fftshift(np.fft.fftfreq(n=padded_movies.x.size, d=psize))
    block_fourier = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded_movies)))
    block_fourier = xr.DataArray(block_fourier, coords=padded_movies.coords)
    block_fourier = block_fourier.rename({'x': 'u', 'y': 'v'}).assign_coords(u=freqs, v=freqs)

    if obs is None:
        # Extra phase to match centroid convention
        phase = np.exp(-1j * np.pi * psize * ((1 + movies.x.size % 2) * block_fourier.u +
                                              (1 + movies.y.size % 2) * block_fourier.v))

        # Pulse function
        pulsefac = trianglePulse_F(2 * np.pi * block_fourier.u, psize) * \
                   trianglePulse_F(2 * np.pi * block_fourier.v, psize)

        block_vis = block_fourier * phase * pulsefac

    else:
        obslist = obs.tlist()
        u = np.concatenate([obsdata['u'] for obsdata in obslist])
        v = np.concatenate([obsdata['v'] for obsdata in obslist])
        t = np.concatenate([obsdata['time'] for obsdata in obslist])

        if conjugate:
            t = np.concatenate((t, t))
            u = np.concatenate((u, -u))
            v = np.concatenate((v, -v))
            sort_idx = np.argsort(t)
            t = t[sort_idx]
            u = u[sort_idx]
            v = v[sort_idx]

        # Extra phase to match centroid convention
        phase = np.exp(-1j * np.pi * psize * ((1 + movies.x.size % 2) * u + (1 + movies.y.size % 2) * v))

        # Pulse function
        pulsefac = trianglePulse_F(2 * np.pi * u, psize) * trianglePulse_F(2 * np.pi * v, psize)

        block_fourier = block_fourier.assign_coords(
            u2=('u', range(block_fourier.u.size)),
            v2=('v', range(block_fourier.v.size))
        )
        tuv2 = np.vstack((block_fourier.v2.interp(v=v), block_fourier.u2.interp(u=u)))
        if 't' in block_fourier.coords:
            block_fourier = block_fourier.assign_coords(t2=('t', range(block_fourier.t.size)))
            tuv2 = np.vstack((tuv2, block_fourier.t2.interp(t=t)))

        if (block_fourier.ndim == 2) or (block_fourier.ndim == 3):
            visre = nd.map_coordinates(np.ascontiguousarray(np.real(block_fourier).data), tuv2)
            visim = nd.map_coordinates(np.ascontiguousarray(np.imag(block_fourier).data), tuv2)
            vis = visre + 1j * visim
            block_vis = xr.DataArray(vis * phase * pulsefac, dims='index')

        elif block_fourier.ndim == 4:
            # Sample block Fourier on tuv coordinates of the observations
            # Note: using np.ascontiguousarray seems to preform ~twice as fast
            num_block = block_fourier.shape[0]
            num_vis = len(t)
            block_vis = np.empty(shape=(num_block, num_vis), dtype=np.complex128)
            for i, fourier in enumerate(block_fourier):
                visre = nd.map_coordinates(np.ascontiguousarray(np.real(fourier).data), tuv2)
                visim = nd.map_coordinates(np.ascontiguousarray(np.imag(fourier).data), tuv2)
                vis = visre + 1j * visim
                block_vis[i] = vis * phase * pulsefac

            block_vis = xr.DataArray(block_vis, dims=[movies.dims[0], 'index'],
                                     coords={movies.dims[0]: movies.coords[movies.dims[0]],
                                             'index': range(num_vis)})
        else:
            raise ValueError("unsupported number of dimensions: {}".format(block_fourier.ndim))

        block_vis.attrs.update(source=obs.source)
        block_vis = block_vis.assign_coords(t=('index', t), u=('index', v), v=('index', u))
    block_vis.attrs.update(movies.attrs)
    block_vis.attrs.update(psize=psize, fft_pad_factor=fft_pad_factor)
    return block_vis

def trianglePulse_F(omega, pdim):
    """
    Modification of eht-imaging trianglePulse_F for an array of points

    Parameters
    ----------
    omega: np.array,
        array of frequencies
    pdim: float,
        pixel size in radians

    References:
    ----------
    https://github.com/achael/eht-imaging/blob/50e728c02ef81d1d9f23f8c99b424705e0077431/ehtim/observing/pulses.py#L91
    """
    pulse = (4.0/(pdim**2 * omega**2)) * (np.sin((pdim * omega)/2.0))**2
    pulse[omega==0] = 1.0
    return pulse

def hdf5_to_xarray(movie):
    frame0 = movie.get_frame(0)
    movie = xr.DataArray(
        data=movie.iframes.reshape(movie.nframes, movie.xdim, movie.ydim),
        coords={'t': movie.times,
                'x': np.linspace(-frame0.fovx()/2, frame0.fovx()/2, frame0.xdim),
                'y': np.linspace(-frame0.fovy()/2, frame0.fovy()/2, frame0.ydim)},
        dims=['t', 'x', 'y'],
        attrs={
            'mjd': movie.mjd,
            'ra': movie.ra,
            'dec': movie.dec,
            'rf': movie.rf
        })
    return movie