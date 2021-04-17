import xarray as _xr
import numpy as _np
import math as _math
from ipywidgets import fixed as _fixed, interactive as _interactive
from IPython.display import display as _display
import h5py as _h5py
import os as _os
import warnings as _warnings
from pathlib import Path as _Path
import inspect as _inspect
from dask.diagnostics import ProgressBar as _ProgressBar
from functools import wraps as _wraps

def linspace_2d(num, start=(-0.5, -0.5), stop=(0.5, 0.5), endpoint=(True, True), units='unitless'):
    """
    Return a 2D DataArray with coordinates spaced over a specified interval.

    Parameters
    ----------
    num: int or tuple
        Number of grid points. If num is a scalar the 2D coordinates are assumed to
        have the same number of points.
    start: (float, float)
        (x, y) starting grid point (included in the grid)
    stop: (float, float)
        (x, y) ending grid point (optionally included in the grid)
    endpoint: (bool, bool)
        Optionally include the stop points in the grid.
    units: str, default='unitless'
        Store the units of the underlying grid.

    Returns
    -------
    grid: xr.DataArray
        A DataArray with coordinates linearly spaced over the desired interval

    Notes
    -----
    Also computes image polar coordinates (r, theta).
    """
    num = (num, num) if _np.isscalar(num) else num
    y = _np.linspace(start[0], stop[0], num[0], endpoint=endpoint[0])
    x = _np.linspace(start[1], stop[1], num[1], endpoint=endpoint[1])
    grid = _xr.Dataset(coords={'y': y, 'x': x})
    grid.y.attrs.update(units=units)
    grid.x.attrs.update(units=units)
    return grid.utils_polar.add_coords()

def full_like(coords, fill_value):
    """
    Return a homogeneous DataArray with specified fill_value.

    Parameters
    ----------
    coords: xr.Coordinates
        xr.Coordinates object specifying the coordinates.
    fill_value: float,
        Homogeneous fill value

    Returns
    -------
    array: xr.DataArray
        A homogeneous DataArray with specified fill_value
    """
    array = _xr.DataArray(coords=coords).fillna(fill_value)
    return array

def matern_sample(coords, length_scale):
    """
    Return a random DataArray sampled from Matern covariance.

    Parameters
    ----------
    coords: xr.Coordinates
        xr.Coordinates object specifying the coordinates.
    length_scale : float or array with shape (yscale, xscale)
        If a float, an isotropic kernel is used. If an array, an anisotropic kernel is used where each dimension
        of defines the length-scale of the respective feature dimension (y, x).

    Returns
    -------
    array: xr.DataArray
        A sampled DataArray

    References
    ----------
    url: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html

    Notes
    -----
    Sampling from high-dimensional matrices resulting from high resolution grids is time consuming.
    For example a grid of 64x64 point makes a 4096x4096 covariance matrix which takes ~20 seconds with 24 CPU cores.
    """
    covariance = matern_covariance(coords['y'], coords['x'], length_scale)
    sample = _np.random.multivariate_normal(_np.zeros(coords['y'].size * coords['x'].size), covariance).reshape(
        (coords['y'].size, coords['x'].size))
    array = _xr.DataArray(data=sample, coords=coords, dims=['y', 'x'],
                          attrs={'desc': 'Matern covariance sampled data',
                                'length_scale': length_scale})
    return array

def matern_covariance(y, x, length_scale):
    """
    Generate a 2D Matern covariance.

    Parameters
    ----------
    y: np.array
        An array of y-coordinates.
    x: np.array
        An array of x-coordinates.
    length_scale : float or array with shape (yscale, xscale)
        If a float, an isotropic kernel is used. If an array, an anisotropic kernel is used where each dimension
        of defines the length-scale of the respective feature dimension (y, x).

    Returns
    -------
    covariance: np.array
        A Matern covariance matrix of size (nx*ny, nx*ny).

    References
    ----------
    url: https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html
    """
    from sklearn.gaussian_process.kernels import Matern
    kernel = Matern(length_scale=length_scale)
    xx, yy = _np.meshgrid(x, y, indexing='xy')
    covariance = kernel(_np.array([yy.ravel(), xx.ravel()]).T)
    return covariance

def load_grmhd(filepath):
    """
    Load GRMHD movie frames (.h5 file)

    Parameters
    ----------
    filepath: str
        path to .h5 file

    Returns
    -------
    movie: xr.DataArray
        GRMHD movie as an xarray object with dims=['t', 'x', 'y'] and coordinates set to:
            't': np.linspace(0, 1, nt)
            'y': np.linspace(-0.5, 0.5, ny)
            'x': np.linspace(-0.5, 0.5, nx)
    """
    filename =  _os.path.abspath(filepath).split('/')[-1][:-3]
    with _h5py.File(filepath, 'r') as file:
        frames = file['I'][:]
    nt, ny, nx = frames.shape
    movie = _xr.DataArray(data=frames,
                          coords={'t': _np.linspace(0, 1, nt),
                                 'y': _np.linspace(-0.5, 0.5, ny),
                                 'x': _np.linspace(-0.5, 0.5, nx)},
                          dims=['t', 'y', 'x'],
                          attrs={'GRMHD': filename})
    return movie

def slider_select_file(dir, filetype=None):
    """
    Slider for interactive selection of a file

    Parameters
    ----------
    dir: str,
        The directory from which to choose.
    filetype:  str, optional,
        Filetypes to display. If filetype is None then all files in the directory are displayed.

    Returns
    -------
    file: interactive,
        An interactive slider utility.
    """

    def select_path(i, paths):
        print(paths[i])
        return paths[i]

    filetype = '*' if filetype is None else '*.' + filetype
    paths = [str(path) for path in _Path(dir).rglob('{}'.format(filetype))]
    file = _interactive(select_path, i=(0, len(paths)-1), paths=_fixed(paths));
    _display(file)
    return file

def next_power_of_two(x):
    """
    Find the next greatest power of two

    Parameters
    ----------
    x: int,
        Input integer

    Returns
    -------
    y: int
       Next greatest power of two
    """
    y = 2 ** (_math.ceil(_math.log(x, 2)))
    return y

@_xr.register_dataarray_accessor("fourier")
class _FourierAccessor(object):
    """
    Register a custom accessor FourierAccessor on xarray.DataArray object.
    This adds methods Fourier manipulations of data and coordinates of movies and images.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def check_image_dims(func):
        """
        A decorator (wrapper) for observations to check that 'x', 'y' exist and are equal

        Returns
        -------
        wrapper: pynoisy.observation.check_xy_equal,
        """

        @_wraps(func)
        def wrapper(*args, **kwargs):
            kwargs = aggregate_kwargs(func, *args, **kwargs)
            if (kwargs['image_dims'][0] not in args[0]._obj.dims) or (kwargs['image_dims'][1] not in args[0]._obj.dims):
                raise AttributeError('DataArray needs both image coordinate: {},{}'.format(*kwargs['image_dims']))
            _np.testing.assert_equal(args[0]._obj[kwargs['image_dims'][0]].data,
                                     args[0]._obj[kwargs['image_dims'][1]].data)
            output = func(**kwargs)
            return output

        return wrapper

    def _extra_phase(self, u, v, psize, image_dim_sizes):
        """
        Extra phase to match centroid convention.

        Parameters
        ----------
        u, v: np.arrays or xr.DataArray
            arrays of Fourier frequencies.
        psize: float,
            pixel size
        image_dim_sizes: tuple (image_dim0.size, image_dim1.size)
            Image dimension sizes (without padding)

        Returns:
        ----------
        phase: xr.DataArray
            A 2D phase function
        """
        phase = _np.exp(-1j * _np.pi * psize * ((1 + image_dim_sizes[0] % 2) * u +
                                                (1 + image_dim_sizes[1] % 2) * v))
        return phase

    def _trianglePulse2D(self, u, v, psize):
        """
        Modification of eht-imaging trianglePulse_F for a DataArray of points

        Parameters
        ----------
        u, v: np.arrays or xr.DataArray
            arrays of Fourier frequencies.
        psize: float,
            pixel size

        Returns
        -------
        pulse: xr.DataArray
            A 2D triangle pulse function

        References:
        ----------
        https://github.com/achael/eht-imaging/blob/50e728c02ef81d1d9f23f8c99b424705e0077431/ehtim/observing/pulses.py#L91
        """
        pulse_u = (4.0 / (psize ** 2 * (2 * _np.pi * u) ** 2)) * (_np.sin((psize * (2 * _np.pi * u)) / 2.0)) ** 2
        pulse_u[2 * _np.pi * u == 0] = 1.0

        pulse_v = (4.0 / (psize ** 2 * (2 * _np.pi * v) ** 2)) * (_np.sin((psize * (2 * _np.pi * v)) / 2.0)) ** 2
        pulse_v[2 * _np.pi * v == 0] = 1.0

        return pulse_u * pulse_v

    @check_image_dims
    def fft(self, fft_pad_factor=2, image_dims=['y', 'x'], fft_dims=['u', 'v']):
        """
        Fast Fourier transform of one or several movies.
        Fourier is done per each time slice on image dimensions

        Parameters
        ----------
        fft_pad_factor: float, default=2
            A padding factor for increased fft resolution.
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.
        fft_dims: [dim1, dim2], default=['u', 'v']
            Fourier data dimensions.

        Returns
        -------
        fft: xr.DataArray,
            A DataArray with the transformed signal.

        Notes
        -----
        For high order downstream interpolation (e.g. cubic) of uv points a higher fft_pad_factor should be taken.
        """
        movies = self._obj.squeeze()

        # Pad images according to pad factor (interpolation in Fourier space)
        ny, nx = movies[image_dims[0]].size, movies[image_dims[1]].size
        npad = next_power_of_two(fft_pad_factor * _np.max((nx, ny)))
        padvalx1 = padvalx2 = int(_np.floor((npad - nx) / 2.0))
        padvaly1 = padvaly2 = int(_np.floor((npad - ny) / 2.0))
        padvalx2 += 1 if nx % 2 else 0
        padvaly2 += 1 if ny % 2 else 0
        padded_movies = movies.pad({image_dims[0]: (padvaly1, padvaly2),
                                    image_dims[1]: (padvalx1, padvalx2)}, constant_values=0.0)

        # Compute visibilities (Fourier transform) of the entire block
        psize = movies.utils_image.psize
        freqs = _np.fft.fftshift(_np.fft.fftfreq(n=padded_movies[image_dims[0]].size, d=psize))
        fft = _np.fft.fftshift(_np.fft.fft2(_np.fft.ifftshift(padded_movies)))

        coords = padded_movies.coords.to_dataset()
        for image_dim, fft_dim in zip(image_dims, fft_dims):
            coords = coords.swap_dims({image_dim: fft_dim}).drop(image_dim).assign_coords({fft_dim: freqs})
            coords[fft_dim].attrs.update(inverse_units=movies[image_dim].units)
        fft = _xr.DataArray(data=fft, dims=coords.dims, coords=coords.coords)
        return fft

    @property
    def phase(self):
        return _xr.DataArray(_np.angle(self._obj.data),
                             coords=self._obj.coords,
                             dims=self._obj.dims,
                             attrs=self._obj.attrs)
    @property
    def logmagnitude(self):
        return _xr.DataArray(_np.log(_np.abs(self._obj.data)),
                             coords=self._obj.coords,
                             dims=self._obj.dims,
                             attrs=self._obj.attrs)

@_xr.register_dataarray_accessor("utils_movie")
class _MovieAccessor(object):
    """
    Register a custom accessor MovieAccessor on xarray.DataArray object.
    This adds methods for 3D manipulation of data and coordinates of movies (should have dimension 't').
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def set_time(self, tstart, tstop, units='UTC hr'):
        movie = self._obj.assign_coords(t=_np.linspace(tstart, tstop, self._obj.t.size))
        movie.t.attrs.update(units=units)
        return movie

    def log_perturbation(self, flux_threshold=1e-15):
        """
        Compute the dynamic perturbation part of a movie
            video = envelope * exp(grf) ---> grf = log(movie) - E_t(log(movie))

        Parameters
        ----------
        flux_threshold: float, default=1e-10
            Minimal flux value to avoid taking a logarithm of zero.

        Returns
        -------
        log_perturbation: xr.DataArray or xr.Dataset
            The logarithm of the data with the static part removed by subtracting mean('t').
        """
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            log_movie = _np.log(self._obj.where(self._obj > flux_threshold))
        log_perturbation = log_movie - log_movie.mean('t')
        log_perturbation.attrs.update(self._obj.attrs)
        return log_perturbation

    @property
    def lightcurve(self):
        """
        The lightcurve of the movie

        Returns
        -------
        lightcurve: xr.DataArray
            The lightcurve of the underlying movie

        Notes
        -----
        Assumes image dimensions to be ['y', 'x']
        """
        lightcurve = self._obj.sum(('y', 'x'))
        return lightcurve

@_xr.register_dataset_accessor("utils_image")
@_xr.register_dataarray_accessor("utils_image")
class _ImageAccessor(object):
    """
    Register a custom accessor ImageAccessor on xarray.DataArray and xarray.Dataset objects.
    This adds methods for 2D manipulation of data and coordinates.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def set_fov(self, fov, image_dims=['y', 'x'],):
        """
        Change the field of view of the underlying image data.

        Parameters
        ----------
        fov: tuple (float, str)
            Field of view and units.
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.
        units: str, default='uas'
            Image domain units. Default is micro arc seconds.

        Returns
        -------
        data: xr.DataArray or xr.Dataset
            A DataArray or Dataset with the underlying 2D coordinates updated.

        Notes
        -----
        Assumes a symmetric fov across image dimensions
        """
        data = self._obj
        fov, units = fov
        grid = linspace_2d((data[image_dims[0]].size, data[image_dims[1]].size),
                           (-fov / 2.0, -fov / 2.0), (fov / 2.0, fov / 2.0),
                           endpoint=(True, True), units=units)
        for dim in image_dims:
            data = data.assign_coords({dim: grid[dim]})

        # Add polar coordinates if exist
        if ('r' in data.coords) or ('theta' in data.coords):
            data = data.utils_polar.add_coords()
        return data

    def regrid(self, num, image_dims=['y', 'x'], method='linear'):
        """
        Re-grid (resample) image dimensions.

        Parameters
        ----------
        num: int or tuple (ny, nx)
            Number of grid points. If num is a scalar the 2D coordinates are assumed to
            have the same number of points.
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.
        method : str, default='linear'
            The method used to interpolate. Choose from {'linear', 'nearest'} for multidimensional array.

        Returns
        -------
        output: xr.DataArray or xr.Dataset
            A DataArray or Dataset regridded with a new spatial resolution.
        """
        units = self._obj[image_dims[0]].units if 'units' in self._obj[image_dims[0]].attrs else 'unitless'
        grid = linspace_2d(num, (self._obj[image_dims[0]][0], self._obj[image_dims[1]][0]),
                           (self._obj[image_dims[0]][-1], self._obj[image_dims[1]][-1]))
        output = self._obj.interp_like(grid,  method=method)
        for dim in image_dims:
            output[dim].attrs.update(units=units)
        return output

    @property
    def psize(self, dim='y'):
        """
        Get the pixel size from the underlying image data Coordinates.

        Parameters
        ----------
        dim: str, default='y',
            Image data dimension.

        Returns
        -------
        psize: float
            The pixel size of the underlying DataArray

        Notes
        -----
        Assumes a symmetric fov across image dimensions
        """
        psize = self._obj.utils_image.fov[0] / self._obj[dim].size
        return float(psize)

    @property
    def fov(self, dim='y'):
        """
        Get the field of view from the underlying image data Coordinates.

        Parameters
        ----------
        dim: str, default='y',
            Image data dimension.

        Returns
        -------
        fov: tuple (float, str),
            The field of view and units of the underlying DataArray

        Notes
        -----
        Assumes a symmetric fov across image dimensions
        """
        fov = self._obj[dim][-1] - self._obj[dim][0]
        return (float(fov), self._obj[dim].units)

@_xr.register_dataset_accessor("utils_polar")
@_xr.register_dataarray_accessor("utils_polar")
class _PolarAccessor(object):
    """
    Register a custom accessor PolarAccessor on xarray.DataArray and xarray.Dataset objects.
    This adds methods for polar coordinate processing on a 2D (x,y) grid.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def add_coords(self, image_dims=['y', 'x']):
        """
        Add polar coordinates to image_dimensions

        Parameters
        ----------
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.

        Returns
        -------
        grid: xr.DataArray,
            A data array with additional polar coordinates 'r' and 'theta'
        """
        if not(image_dims[0] in self._obj.coords and image_dims[1] in self._obj.coords):
            raise AttributeError('Coordinates have to contain both x and y')
        x, y = self._obj[image_dims[0]], self._obj[image_dims[1]]
        yy, xx = _np.meshgrid(y, x, indexing='ij')
        r = _np.sqrt(xx ** 2 + yy ** 2)
        theta = _np.arctan2(yy, xx)
        grid = self._obj.assign_coords({'r': ([image_dims[0], image_dims[1]], r),
                                        'theta': ([image_dims[0], image_dims[1]], theta)})

        # Add units to 'r' and 'theta'
        units = None
        if ('units' in grid[image_dims[0]].attrs) and ('units' in grid[image_dims[1]].attrs):
            units = 'units'
        elif ('inverse_units' in grid[image_dims[0]].attrs) and ('inverse_units' in grid[image_dims[1]].attrs):
            units = 'inverse_units'

        if units is not None:
            if grid[image_dims[0]].attrs[units] != grid[image_dims[1]].attrs[units]:
                raise AttributeError('different units for x and y not supported')
            grid['r'].attrs.update({units: grid[image_dims[0]].attrs[units]})
        grid['theta'].attrs.update(units='rad')
        return grid

    def r_cutoff(self, r0, fr0, dfr0, f0):
        """
        Smooth cutoff at radius r0 (continuous + once differentiable at r0).

        Parameters
        ----------
        r0: float,
            Cut-off radius
        fr0: float,
            Function value at r0: f(r0)
        dfr0: float,
            Function slope at r0: df(r0)
        f0: float,
            Function value at r=0: f(0)

        References
        ----------
        https://github.com/aviadlevis/inoisy/blob/47fb41402ecdf93bfdd176fec780e8f0ba43445d/src/param_general_xy.c#L97
        """
        output = self._obj if 'r' in self._obj.coords else self._obj.utils_polar.add_coords()
        r = output['r']
        b = (2. * (fr0 - f0) - r0 * dfr0) / r0 ** 3
        a = (fr0 - f0) / (b * r0 * r0) + r0
        return b * r * r * (a - r) + f0

    def w_keplerian(self, r_cutoff):
        """
        Compute Keplerian orbital frequency as a function of radius.

        Parameters
        ----------
        r_cutoff: float
            Cutoff radius for a smooth center point.

        Returns
        -------
        w: xr.DataArray
            A DataArray of Keplerian orbital frequency on an x,y grid.
        """
        if 'r' not in self._obj.coords:
            self._obj.utils_polar.add_coords()
        r = self._obj['r']
        w = r ** (-1.5)
        if r_cutoff > 0.0:
            w.values[(r < r_cutoff).data] = w.utils_polar.r_cutoff(r_cutoff, r_cutoff ** (-1.5), -1.5 * r_cutoff ** (-2.5),
                                                             0.9 * r_cutoff ** (-1.5)).values[(r < r_cutoff).data]
        w.name = 'w_keplerian'
        w.attrs.update(r_cutoff=r_cutoff)
        return w

@_xr.register_dataset_accessor("utils_tensor")
class _TensorAccessor(object):
    """
    Register a custom accessor TensorAccessor on xarray.Dataset object.
    This adds properties and methods for processing the diffusion tensor fields.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def set_v_magnitude(self, magnitude):
        """
        Set a new velocity field magnitude.

        Parameters
        ----------
        magnitude: scalar or xr.DataArray
            New magnitude for the velocity field.  magnitude = sqrt(vx** 2 + vy**2).

        Returns
        -------
        advection: xr.Dataset
            A Dataset with velocity (vx, vy) on an x,y grid.
        """
        previous_magnitude = self._obj.utils_tensor.v_magnitude
        self._obj['vy'] = (magnitude/previous_magnitude) * self._obj['vy']
        self._obj['vx'] = (magnitude/previous_magnitude) * self._obj['vx']
        return self._obj


    @property
    def diffusion_coefficient(self, threshold=1e-8):
        """
        Compute diffusion coefficient:
            2*correlation_length** 2 / correlation_time.clip(threshold).

        Parameters
        ----------
        threshold: float
            Minimal correlation time to avoid division by zero.

        Returns
        -------
        diffusion_coefficient: xr.DataArray
            A DataArray diffusion coefficients on an x,y grid.
        """
        if 'correlation_time' in self._obj and 'correlations_length' in self._obj:
            return 2 * self._obj.correlation_length ** 2 / self._obj.correlation_time.clip(threshold)
        else:
            raise AttributeError('Dataset has to contain both correlations_length and correlation_time')

    @property
    def v_magnitude(self):
        """
        Compute velocity field magnitude:
           v_magnitude = sqrt(vx** 2 + vy**2).

        Returns
        -------
        v_magnitude: xr.DataArray
            A DataArray with velocity magnitude on an x,y grid.
        """
        if 'vx' in self._obj and 'vy' in self._obj:
            return _np.sqrt(self._obj.vx ** 2 + self._obj.vy ** 2)
        else:
            raise AttributeError('Dataset has to contain both vx and vy')

    @property
    def v_angle(self):
        """
        Compute velocity field angle:
           v_angle = arctan(vy/vx).

        Returns
        -------
        v_angle: xr.DataArray
            A DataArray with velocity angle on an x,y grid.
        """
        if 'vx' in self._obj and 'vy' in self._obj:
            return _np.arctan2(self._obj.vy, self._obj.vx)
        else:
            raise AttributeError('Dataset has to contain both vx and vy')

def aggregate_kwargs(func, *args, **kwargs):
    """
    Update kwargs dictionary with args and defaults parameters for func
    """
    signature = _inspect.signature(func)
    kwargs.update(
        {k: v.default for k, v in signature.parameters.items() if v.default is not _inspect.Parameter.empty}
    )
    args_keys = list(signature.parameters.keys())
    for i, arg in enumerate(args):
        kwargs.update({args_keys[i]: arg})
    return kwargs

def mode_map(output_type, data_vars=None, progress_bar=True):
    """
    A decorator (wrapper) for dask computations over the entire mode dataset for an output a DataArray or Dataset.

    Parameters
    ----------
    output_type: 'DataArray' or 'Dataset'
        Two xarray output types
    data_dars: string or list of strings, optional
        If the output_type is Dataset, the string/list of strings should specify the data_vars
    progress_bar: bool, default=True,
        Progress bar is useful as manifold computations can be time consuming.

    Returns
    -------
    wrapper: pynoisy.utils.mode_map,
        A wrapped function which takes care of dask related tasks with some pre processing.
    """
    def decorator(func):
        @_wraps(func)
        def wrapper(modes, *args, **kwargs):
            # Manifold dimensions are the modes dimensions without ['degree', 't', 'x', 'y'].
            coords = modes.coords.to_dataset()
            for dim in ['degree', 't', 'x', 'y']:
                del coords[dim]
            dim_names = list(coords.dims.keys())
            dim_sizes = list(coords.dims.values())

            # Generate an output template for dask which fits in a single chunk
            if (output_type == 'Dataset'):
                if data_vars is None:
                    raise AttributeError('For output_type="Dataset" data_vars arguments should be specified')

                data_dict = dict()
                if isinstance(data_vars, str):
                    data_dict[data_vars] = (dim_names, _np.empty(dim_sizes))
                elif isinstance(data_vars, list):
                    for var_name in data_vars:
                        data_dict[var_name] = (dim_names, _np.empty(dim_sizes))
                else:
                    raise AttributeError('data_vars can be either string or list of strings')

                template = _xr.Dataset(data_vars=data_dict, coords=coords.coords).chunk(
                    dict(zip(dim_names, [1] * len(dim_names))))

            elif (output_type == 'DataArray'):
                template = _xr.DataArray(coords=coords.coords).chunk(
                    dict(zip(dim_names, [1] * len(dim_names))))

            else:
                raise AttributeError('data types allowed are: "DataArray" or "Dataset"')

            # Generate dask computation graph
            mapped = _xr.map_blocks(func, modes, template=template, args=args, kwargs=kwargs)

            if progress_bar:
                with _ProgressBar():
                    output = mapped.compute()
            else:
                output = mapped.compute()
            return output
        return wrapper

    return decorator


#########
"""
def sample_eht(fourier, obs, conjugate=False, format='array', method='linear'):
    obslist = obs.tlist()
    u = np.concatenate([obsdata['u'] for obsdata in obslist])
    v = np.concatenate([obsdata['v'] for obsdata in obslist])
    t = np.concatenate([obsdata['time'] for obsdata in obslist])

    if conjugate:
        t = np.concatenate((t, t))
        u = np.concatenate((u, -u))
        v = np.concatenate((v, -v))

    if format == 'movie':
        # For static images duplicate temporal dimension
        if 't' not in fourier.coords:
            obstimes = [obsdata[0]['time'] for obsdata in obslist]
            fourier = fourier.expand_dims(t=obstimes)

        data = fourier.sel(
            t=xr.DataArray(t, dims='index'),
            u=xr.DataArray(v, dims='index'),
            v=xr.DataArray(u, dims='index'), method='nearest'
        )
        eht_measurements = xr.full_like(fourier, fill_value=np.nan)
        eht_measurements.attrs.update(fourier.attrs)
        eht_measurements.attrs.update(source=obs.source)
        eht_measurements.loc[{'t': data.t, 'u': data.u, 'v': data.v}] = data

    elif format == 'array':
        eht_coords = {'u': xr.DataArray(v, dims='index'), 'v': xr.DataArray(u, dims='index')}
        if 't' in fourier.coords:
            eht_coords.update(t=xr.DataArray(t, dims='index'))
        eht_measurements = fourier.interp(eht_coords, method=method)
    else:
        raise NotImplementedError('format {} not implemented'.format(format))

    return eht_measurements

def opening_angles_vis_residuals(files, measurements, obs, envelope, interp_method='linear', damp=0.0, degree=np.inf,
                                 fft_pad_factor=1, return_coefs=False, conjugate=False):

    fov = float(envelope['x'].max() - envelope['x'].min())
    envelope_fourier = ehtf.compute_block_visibilities(envelope, measurements.psize, fft_pad_factor=fft_pad_factor,
                                                       conjugate=conjugate)
    envelope_eht = sample_eht(envelope_fourier, obs, conjugate=conjugate, method=interp_method)
    dynamic_measurements = measurements - envelope_eht
    dynamic_measurements.attrs.update(measurements.attrs)
    measurements = dynamic_measurements

    residuals, coefficients = [], []
    for file in tqdm(files):
        modes = pynoisy.utils.load_modes(file, dtype=float).noisy_methods.to_world_coords(
            tstart=float(measurements.t[0]), tstop=float(measurements.t[-1]), fov=fov)

        residual, coefs = [], []
        degree = min(degree, modes.deg.size)
        for temporal_angle in tqdm(modes.temporal_angle, leave=False):
            modes_reduced = modes.sel(temporal_angle=temporal_angle).isel(deg=slice(degree)).dropna('deg')
            movies = modes_reduced.eigenvectors * envelope if envelope is not None else modes_reduced.eigenvectors
            movies = movies * modes_reduced.eigenvalues

            # Split subspace for large fft_pad_factor memory req
            subspace = []
            slice_size = int( np.ceil(movies.deg.size / fft_pad_factor))
            for i in range(fft_pad_factor):
                modes_fourier = ehtf.compute_block_visibilities(
                    movies=movies.isel(deg=slice(i*slice_size, (i+1)*slice_size)),
                    psize=measurements.psize, fft_pad_factor=fft_pad_factor)
                modes_eht = sample_eht(modes_fourier, obs, conjugate=conjugate, method=interp_method)
                subspace.append(modes_eht)
                del modes_fourier
            subspace = xr.concat(subspace, dim='deg')

            output = pynoisy.linalg.projection_residual(measurements, subspace, damp=damp, return_coefs=return_coefs)

            if return_coefs:
                res, coef = output
                coefs.append(coef.expand_dims(spatial_angle=modes.spatial_angle, temporal_angle=[temporal_angle]))
            else:
                res = output
            residual.append(res.expand_dims(spatial_angle=modes.spatial_angle, temporal_angle=[temporal_angle]))

        residual = xr.concat(residual, dim='temporal_angle')
        if return_coefs:
            coefficients.append(xr.concat(coefs, dim='temporal_angle'))
        residuals.append(residual)
        del modes
        gc.collect()

    residuals = xr.concat(residuals, dim='spatial_angle').sortby('spatial_angle').expand_dims(deg=[degree])
    residuals.attrs.update(measurements.attrs)
    residuals.attrs.update(file_num=len(files), damp=damp,
                           interp_method=interp_method, fov=fov, conjugate=str(conjugate),
                           with_envelope='False' if envelope is None else 'True')
    output = residuals
    if return_coefs:
        coefficients = xr.concat(coefficients, dim='spatial_angle').sortby('spatial_angle')
        coefficients.attrs.update(residuals.attrs)
        output = (residuals, coefficients)

    return output
"""

def save_complex(dataset, *args, **kwargs):
    ds = dataset.expand_dims('reim', axis=-1) # Add ReIm axis at the end
    ds = _xr.concat([ds.real, ds.imag], dim='reim')
    return ds.to_netcdf(*args, **kwargs)

def read_complex(*args, **kwargs):
    ds = _xr.open_dataset(*args, **kwargs)
    output = ds.isel(reim=0) + 1j * ds.isel(reim=1)
    output.attrs.update(ds.attrs)
    output = output.to_array().squeeze('variable') if len(output.data_vars.items()) == 1 else output
    return output


