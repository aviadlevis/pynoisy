"""
Utility functions and methods.
"""
import xarray as _xr
import numpy as _np
import math as _math
from ipywidgets import fixed as _fixed, interactive as _interactive
from IPython.display import display as _display
import h5py as _h5py
import os as _os
import gc as _gc
import warnings as _warnings
from pathlib import Path as _Path
import inspect as _inspect
from dask.diagnostics import ProgressBar as _ProgressBar
from functools import wraps as _wraps
import subprocess as _subprocess

def linspace_2d(num, start=(-0.5, -0.5), stop=(0.5, 0.5), endpoint=(True, True), units='unitless'):
    """
    Return a 2D DataArray with coordinates spaced over a specified interval.

    Parameters
    ----------
    num: int or tuple
        Number of grid points in (y, x) dimensions. If num is a scalar the 2D coordinates are assumed to
        have the same number of points.
    start: (float, float)
        (y, x) starting grid point (included in the grid)
    stop: (float, float)
        (y, x) ending grid point (optionally included in the grid)
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

def mode_map(output_type, data_vars=None, out_dims=None, chunk=1, progress_bar=True):
    """
    A decorator (wrapper) for dask computations over the entire mode dataset for an output a DataArray or Dataset.

    Parameters
    ----------
    output_type: 'DataArray' or 'Dataset'
        Two xarray output types
    data_dars: string or list of strings, optional
        If the output_type is Dataset, the string/list of strings should specify the data_vars
    out_dims: string or list of strings, optional
        If None, the output dimensions are assumed to be the dataset dimensions which are not ['degree', 't', 'x', 'y'].
    chunk: int or list, default=1,
        Output chunks size along the out_dims dimensions. Default=1 which extends along all output dimensions/.
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

            # Drop non-dimensional coordinates
            for dim in modes.coords.keys():
                if dim not in modes.dims:
                    modes = modes.drop(dim)

            # Modes dimensions without ['degree', 't', 'y', 'x'].
            coords = modes.coords.to_dataset()
            if out_dims is None:
                for dim in ['degree', 't', 'y', 'x']:
                    del coords[dim]
                dim_names = list(coords.dims.keys())
                dim_sizes = list(coords.dims.values())

            # User specified output dimensions
            else:
                dim_names = out_dims
                dim_sizes = [modes.dims[dim] for dim in out_dims]
                for dim in coords.dims:
                    if not dim in dim_names:
                        del coords[dim]
            coords = coords.coords

            chunks = chunk
            if _np.isscalar(chunks):
                chunks = [chunks] * len(dim_names)
            else:
                if len(chunks) != len(dim_names):
                    raise AttributeError('Chunks and output dims have different lengths: out_dims={}, chunks={}'.format(
                        dim_names, chunks))

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
                template = _xr.Dataset(data_vars=data_dict, coords=coords).chunk(dict(zip(dim_names, chunks)))

            elif (output_type == 'DataArray'):
                template = _xr.DataArray(coords=coords, dims=dim_names).chunk(dict(zip(dim_names, chunks)))

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

def github_version():
    """
    Get github version.

    Returns
    -------
    github_version: str,
        The current GitHub version.

    Raises
    ------
    warning if there are uncomitted changes in pynoisy or inoisy.
    """
    dirs = ['pynoisy', 'inoisy']
    github_dirs = [_Path(_os.environ['INOISY_DIR']).parent.joinpath(dir) for dir in dirs]
    uncomitted_changes = _subprocess.check_output(["git", "diff", "--name-only", *github_dirs]).strip().decode('UTF-8')
    if uncomitted_changes:
        _warnings.warn('There are uncomitted changes in the pynoisy/inoisy directories: {}'.format(uncomitted_changes))
    github_version = _subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('UTF-8')
    github_version = github_version + ' + uncomitted changes' if uncomitted_changes else github_version
    return github_version

@_xr.register_dataarray_accessor("utils_fourier")
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
        movies = self._obj

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
        fft = _xr.DataArray(fft, coords=padded_movies.coords)

        # Delete padded movies for memory utilization
        del padded_movies
        _gc.collect()

        for image_dim, fft_dim in zip(image_dims, fft_dims):
            fft = fft.swap_dims({image_dim: fft_dim}).drop(image_dim).assign_coords({fft_dim: freqs})
            fft[fft_dim].attrs.update(inverse_units=movies[image_dim].units)
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

    def check_time_units(self, units='UTC'):
        """
        A utility for functions which checks the temporal units.

        Parameters
        ----------
        units: str, default='UTC'
            Temporal units.

        Raises
        ------
        Warning if units of dimension 't' are not the same as units.
        """
        if ('t' in self._obj.coords) and (units != self._obj['t'].units):
            _warnings.warn('Units on dim "t" are not recognized, should be {}'.format(units))

    def set_time(self, tstart, tstop, units='UTC'):
        """
        Set the temporal units of the movie.

        Parameters
        ----------
        tstart: float,
            Start time of the observation in hours
        tstop: float,
            End time of the observation in hours
        units: str, default='UTC'
            Temporal units.

        Returns
        -------
        movie: xr.DataArray
            Movie with new time coordinates set.
        """
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

    def change_units(self, units='rad', image_dims=['y', 'x']):
        """
        A utility for functions which require radian units along image dimensions.

        Parameters
        ----------
        units: str, default='rad'
            Units supported are 'rad', 'uas', 'as'.
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.

        Returns
        -------
        coords: xr.DataArray or xr.Dataset,
            DataArray or Dataset with image_dims coordinates converted to radians.

        Raises
        ------
        Warning if units are not: 'rad', 'uas', 'as'.
        """
        unit_conversion = {
            'rad': {'rad': 1.0, 'as': 4.848136811094136e-06, 'uas': 4.848136811094136e-12},
            'as': {'rad': 206264.80624714843, 'as': 1.0, 'uas': 1e6},
            'uas': {'rad': 206264806247.14844, 'as': 1e-6, 'uas': 1.0}
        }
        output = self._obj.copy()
        for dim in image_dims:
            input_units = output[dim].units
            if (input_units == 'rad') or (input_units == 'as') or (input_units == 'uas') :
                output[dim] = _xr.DataArray(output[dim] * unit_conversion[units][input_units], attrs={'units': units})
            else:
                _warnings.warn('Units not recognized (should be rad/as/uas). Assuming rad units')
        return output

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

@_xr.register_dataset_accessor("utils_io")
@_xr.register_dataarray_accessor("utils_io")
class IOAccessor(object):
    """
    Register a custom accessor MovieAccessor on xarray.DataArray and xarray.Dataset objects.
    This adds methods for input/output saving to disk large dataset and coordinates.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def _expand_variable(self, nc_variable, data, expanding_dim, nc_shape, added_size):
        # For time deltas, we must ensure that we use the same encoding as what was previously stored.
        # We likely need to do this as well for variables that had custom econdings too
        if hasattr(nc_variable, 'calendar'):
            data.encoding = {
                'units': nc_variable.units,
                'calendar': nc_variable.calendar,
            }
        data_encoded = _xr.conventions.encode_cf_variable(data)  # , name=name)
        left_slices = data.dims.index(expanding_dim)
        right_slices = data.ndim - left_slices - 1
        nc_slice = (slice(None),) * left_slices + (slice(nc_shape, nc_shape + added_size),) + (slice(None),) * (
            right_slices)
        nc_variable[nc_slice] = data_encoded.data

    def append_to_netcdf(self, filename, unlimited_dims):
        """
        Append data to existing netcdf file along specified dimensions

        Parameters
        ----------
        filename: str,
            Netcdf file name
        unlimited_dims: str or list of strings,
            Dimension(s) that should be serialized as unlimited dimensions

        References
        ----------
        https://github.com/pydata/xarray/issues/1672#issuecomment-685222909
        """
        import netCDF4

        if isinstance(unlimited_dims, str):
            unlimited_dims = [unlimited_dims]

        if len(unlimited_dims) != 1:
            # TODO: change this so it can support multiple expanding dims
            raise ValueError('One unlimited dim is supported, got {}'.format(len(unlimited_dims)))

        unlimited_dims = list(set(unlimited_dims))
        expanding_dim = unlimited_dims[0]

        with netCDF4.Dataset(filename, mode='a') as nc:
            nc_coord = nc[expanding_dim]
            nc_shape = len(nc_coord)

            added_size = len(self._obj[expanding_dim])
            variables, attrs = _xr.conventions.encode_dataset_coordinates(self._obj)

            for name, data in variables.items():
                if expanding_dim not in data.dims:
                    # Nothing to do, data assumed to the identical
                    continue

                nc_variable = nc[name]
                self._expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size)

    @_wraps(_xr.Dataset.to_netcdf)
    def save_complex(self, *args, **kwargs):
        """
        Wraps NetCDF saving of complex data by appending reim axis at the end

        References
        ----------
        http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_netcdf.html
        """
        ds = self._obj.expand_dims('reim', axis=-1)
        ds = _xr.concat([ds.real, ds.imag], dim='reim')
        return ds.to_netcdf(*args, **kwargs)

@_wraps(_xr.load_dataset)
def read_complex(*args, **kwargs):
    """
    Wraps NetCDF reading of complex data by stacking reim axis

    References
    ----------
    http://xarray.pydata.org/en/stable/generated/xarray.Dataset.to_netcdf.html
    """
    ds = _xr.open_dataset(*args, **kwargs)
    output = ds.isel(reim=0) + 1j * ds.isel(reim=1)
    output.attrs.update(ds.attrs)
    output = output.to_array().squeeze('variable') if len(output.data_vars.items()) == 1 else output
    return output
