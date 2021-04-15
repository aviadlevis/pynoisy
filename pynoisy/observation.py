"""
Observations classes and functions to compute EHT interferometric observations.
Here is where the interface with eht-imaging [1] library resides.

References
----------
.. [1] eht-imaging: https://github.com/achael/eht-imaging
"""
import xarray as xr
import numpy as np
import scipy.ndimage as nd
import warnings
import ehtim as eh
import ehtim.const_def as ehc
from functools import wraps
from pynoisy.utils import aggregate_kwargs

@xr.register_dataarray_accessor("observe")
class ObserveAccessor(object):
    """
    Register a custom accessor ObserveAccessor on xarray.DataArray object.
    This adds methods for interfacing with eht-imaging classes and methods and computing Fourier domain quantities.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def check_time_units(observe_fn):
        """
        A decorator (wrapper) for observations which require time units to be aligned to UTC hour.

        Returns
        -------
        wrapper: pynoisy.observation.check_time_units,
        """
        @wraps(observe_fn)
        def wrapper(*args, **kwargs):
            if ('t' in args[0]._obj.coords) and (args[0]._obj['t'].units != 'UTC hr'):
                warnings.warn('Units on dim "t" are not recognized, should be "UTC hr"')
            output = observe_fn(*args, **kwargs)
            return output
        return wrapper

    def check_image_units(observe_fn):
        """
        A decorator (wrapper) for observations which require radian units along image dimensions

        Returns
        -------
        wrapper: pynoisy.observation.check_image_units,
        """
        @wraps(observe_fn)
        def wrapper(*args, **kwargs):
            # Convert image dimensions into radian units
            kwargs = aggregate_kwargs(observe_fn, *args, **kwargs)
            rad_unit_conversion = {'rad': 1.0, 'as': ehc.RADPERUAS, 'uas': ehc.RADPERUAS}
            for dim in kwargs['image_dims']:
                dim_units = args[0]._obj[dim].units
                if (dim_units == 'rad') or (dim_units == 'as') or (dim_units == 'uas'):
                    args[0]._obj[dim] = xr.DataArray(args[0]._obj[dim] * rad_unit_conversion[dim_units],
                                                     attrs={'units': 'rad'})
                else:
                    warnings.warn(
                        'Units on dim {} not recognized (should be rad/as/uas). Assuming rad units'.format(dim))
            output = observe_fn(**kwargs)
            return output

        return wrapper

    @check_image_units
    def block_fourier(self, fft_pad_factor=2, image_dims=['y', 'x'], fft_dims=['u', 'v']):
        """
         Fast Fourier transform of one or several movies.
         Fourier is done per each time slice on image dimensions.
         This function adds a 2D triangle pulse and phase to match centroid convention.

        Parameters
        ----------
        fft_pad_factor: float
            Padding factor for increased fft resolution.
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.
        fft_dims: [dim1, dim2], default=['u', 'v']
            Fourier data dimensions.

        Returns
        -------
        block_fft: xr.DataArray,
            A data array of shape (num_movies, num_visibilities) and dtype np.complex128.

        Notes
        -----
        Units of input data image_dims must be 'rad', 'as' or uas'.
        For high order downstream interpolation (e.g. cubic) of uv points a higher fft_pad_factor should be taken.
        """
        movies = self._obj
        psize = movies.image.psize
        image_dim_sizes = (movies[image_dims[0]].size, movies[image_dims[1]].size)
        fft = movies.fourier.fft(fft_pad_factor, image_dims, fft_dims)
        fft *= fft.fourier._trianglePulse2D(fft.u, fft.v, psize, fft_dims) * \
               fft.fourier._phase(fft.u, fft.v, psize, image_dim_sizes, fft_dims)

        if ('r' in fft.coords) and ('theta' in fft.coords):
            fft = fft.polar.add_coords(image_dims=['u', 'v'])
        return fft

    @check_time_units
    @check_image_units
    def block_observe_same_nonoise(self, obs, fft_pad_factor=2, image_dims=['y', 'x']):
        """
        Modification of ehtim's observe_same_nonoise method for a block of movies.

        Parameters
        ----------
        obs: Obsdata,
            ehtim observation data containing the array geometry and measurement times.
        fft_pad_factor: float
            Padding factor for increased fft resolution.
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.

        Returns
        -------
        visibilities: xr.DataArray
            DataArray of visibilities, shape = (num_movies, num_visibilities) and dtype np.complex128.

        Notes
        -----
        1. NFFT is not supported.
        2. Cubic interpolation in both time and uv (ehtim has linear interpolation in time and cubic in uv).
        3. Units of input data image_dims must be 'rad', 'as' or uas'.
        4. Due to cubic interpolation of nd.map_coordinates fft_pad_factor should be >= 2.

        References
        ----------
        https://github.com/achael/eht-imaging/blob/6b87cdc65bdefa4d9c4530ea6b69df9adc531c0c/ehtim/movie.py#L981
        https://github.com/achael/eht-imaging/blob/6b87cdc65bdefa4d9c4530ea6b69df9adc531c0c/ehtim/observing/obs_simulate.py#L182
        """
        movies = self._obj
        fft_dims = ['u', 'v']

        image_dim_sizes = (movies[image_dims[0]].size, movies[image_dims[1]].size)
        psize = movies.image.psize

        fft = movies.fourier.fft(fft_pad_factor, image_dims, fft_dims=fft_dims)

        obslist = obs.tlist()
        u = np.concatenate([obsdata['u'] for obsdata in obslist])
        v = np.concatenate([obsdata['v'] for obsdata in obslist])
        t = np.concatenate([obsdata['time'] for obsdata in obslist])

        # Extra phase to match centroid convention
        pulsefac = fft.fourier._trianglePulse2D(u, v, psize, fft_dims)
        phase = fft.fourier._phase(u, v, psize, image_dim_sizes, fft_dims)

        fft = fft.assign_coords(u2=(fft_dims[0], range(fft[fft_dims[0]].size)),
                                v2=(fft_dims[1], range(fft[fft_dims[1]].size)))
        tuv2 = np.vstack((fft.v2.interp({fft_dims[1]: v}), fft.u2.interp({fft_dims[0]: u})))

        if 't' in fft.coords:
            fft = fft.assign_coords(t2=('t', range(fft.t.size)))
            tuv2 = np.vstack((fft.t2.interp(t=t), tuv2))

        if (fft.ndim == 2) or (fft.ndim == 3):
            visre = nd.map_coordinates(np.ascontiguousarray(np.real(fft).data), tuv2)
            visim = nd.map_coordinates(np.ascontiguousarray(np.imag(fft).data), tuv2)
            vis = visre + 1j * visim
            visibilities = xr.DataArray(vis * phase * pulsefac, dims='index')

        elif fft.ndim == 4:
            # Sample block Fourier on tuv coordinates of the observations
            # Note: using np.ascontiguousarray preforms ~twice as fast
            num_block = fft.shape[0]
            num_vis = len(t)
            visibilities = np.empty(shape=(num_block, num_vis), dtype=np.complex128)
            for i, fourier in enumerate(fft):
                visre = nd.map_coordinates(np.ascontiguousarray(np.real(fourier).data), tuv2)
                visim = nd.map_coordinates(np.ascontiguousarray(np.imag(fourier).data), tuv2)
                vis = visre + 1j * visim
                visibilities[i] = vis * phase * pulsefac

            visibilities = xr.DataArray(visibilities,
                                        dims=[movies.dims[0], 'index'],
                                        coords={movies.dims[0]: movies.coords[movies.dims[0]],
                                                'index': range(num_vis)})
        else:
            raise ValueError("unsupported number of dimensions: {}".format(fft.ndim))

        visibilities = visibilities.assign_coords(t=('index', t), u=('index', v), v=('index', u))
        visibilities.attrs.update(fft_pad_factor=fft_pad_factor, source=obs.source)
        return visibilities

    @check_time_units
    @check_image_units
    def to_ehtim(self, ra=17.761121055553343, dec=-29.00784305556, rf=226191789062.5, mjd=57850, source='SGRA',
                 image_dims=['y', 'x']):
        """
        Transform xarray movie to eht-imaging Movie object. Defaults to SgrA*.

        Parameters
        ----------
        ra: float, default=17.761121055553343,
            Source Right Ascension in fractional hours.
        dec: float, default=-29.00784305556,
            Source declination in fractional degrees.
        rf: float, default=226191789062.5,
            Reference frequency observing at corresponding to 1.3 mm wavelength
        mjd: int, default=57850,
            Modified julian date of observation
        source: str, default='SGRA',
            Source name
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.

        Returns
        -------
        output: ehtim.Movie or ehtim.Image objects
            A Movie or Image object depending on the data dimensionality
        """
        movie = self._obj.squeeze()
        if movie.ndim > 3:
            raise AttributeError('Data dimensions are greater than 3.')
        im_list = []
        if 't' in movie.dims:
            for i, time in enumerate(movie.t):
                frame = movie.isel(t=i)
                image = eh.image.make_empty(frame.sizes[image_dims[0]], movie.image.fov[0], ra, dec, rf, mjd=mjd,
                                            source=source)
                image.time = float(time)
                image.ivec = frame.data.ravel()
                im_list.append(image)
            output = eh.movie.merge_im_list(im_list)
        else:
            output = eh.image.make_empty(movie.sizes[image_dims[0]], movie.image.fov[0], ra, dec, rf, mjd=mjd,
                                        source=source)
            output.ivec = movie.data.ravel()
        return output