import pynoisy.linalg
from pynoisy import eht_functions as ehtf
import netCDF4
import xarray as xr
import numpy as np
from ipywidgets import fixed, interactive
from IPython.display import display
import h5py
import os
from tqdm.auto import tqdm
import gc
import warnings

def linspace_2d(num, start=(-0.5, -0.5), stop=(0.5, 0.5), endpoint=(True, True)):
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

    Returns
    -------
    grid: xr.DataArray
        A DataArray with coordinates linearly spaced over the desired interval

    Notes
    -----
    Also computes image polar coordinates (r, theta).
    """
    num = (num, num) if np.isscalar(num) else num
    y = np.linspace(start[0], stop[0], num[0], endpoint=endpoint[0])
    x = np.linspace(start[1], stop[1], num[1], endpoint=endpoint[1])
    grid = xr.Dataset(coords={'y': y, 'x': x}).polar.add_coords()
    return grid

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
    array = xr.DataArray(coords=coords).fillna(fill_value)
    return array

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
        GRMHD movie as an xarray object with dims=['t', 'x', 'y']
    """
    filename =  os.path.abspath(filepath).split('/')[-1][:-3]
    with h5py.File(filepath, 'r') as file:
        frames = file['I'][:]
    nt, nx, ny = frames.shape
    movie = xr.DataArray(data=frames,
                         coords={'t': np.linspace(0, 1, nt),
                                 'y': np.linspace(-0.5, 0.5, nx),
                                 'x': np.linspace(-0.5, 0.5, ny)},
                         dims=['t', 'y', 'x'],
                         attrs={'GRMHD': filename})
    return movie

@xr.register_dataset_accessor("io")
@xr.register_dataarray_accessor("io")
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
        data_encoded = xr.conventions.encode_cf_variable(data)  # , name=name)
        left_slices = data.dims.index(expanding_dim)
        right_slices = data.ndim - left_slices - 1
        nc_slice = (slice(None),) * left_slices + (slice(nc_shape, nc_shape + added_size),) + (slice(None),) * (
            right_slices)
        nc_variable[nc_slice] = data_encoded.data

    def append_to_netcdf(self, filename, unlimited_dims):
        if isinstance(unlimited_dims, str):
            unlimited_dims = [unlimited_dims]

        if len(unlimited_dims) != 1:
            # TODO: change this so it can support multiple expanding dims
            raise ValueError('One unlimited dim is supported, got {}'.format(len(unlimited_dims)))

        unlimited_dims = list(set(unlimited_dims))
        expanding_dim = unlimited_dims[0]

        with netCDF4.Dataset(filename, mode='a') as nc:
            nc_dims = set(nc.dimensions.keys())

            nc_coord = nc[expanding_dim]
            nc_shape = len(nc_coord)

            added_size = len(self._obj[expanding_dim])
            variables, attrs = xr.conventions.encode_dataset_coordinates(self._obj)

            for name, data in variables.items():
                if expanding_dim not in data.dims:
                    # Nothing to do, data assumed to the identical
                    continue

                nc_variable = nc[name]
                self._expand_variable(nc_variable, data, expanding_dim, nc_shape, added_size)


@xr.register_dataset_accessor("movie")
@xr.register_dataarray_accessor("movie")
class MovieAccessor(object):
    """
    Register a custom accessor MovieAccessor on xarray.DataArray and xarray.Dataset objects.
    This adds methods for 3D manipulation of data and coordinates of movies (should have dimension 't').
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def log_perturbation(self, flux_threshold=1e-15):
        """
        Compute the dynamic perturbation part of a movie
            video = envelope * exp(grf) ---> grf = log(movie) - E_t(log(movie))

        Arguments
        ---------
        flux_threshold: float, default=1e-10
            Minimal flux value to avoid taking a logarithm of zero.

        Returns
        -------
        log_perturbation: xr.DataArray or xr.Dataset
            The logarithm of the data with the static part removed by subtracting mean('t').
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_movie = np.log(self._obj.where(self._obj > flux_threshold))
        log_perturbation = log_movie - log_movie.mean('t')
        log_perturbation.attrs.update(self._obj.attrs)
        return log_perturbation

@xr.register_dataset_accessor("image")
@xr.register_dataarray_accessor("image")
class ImageAccessor(object):
    """
    Register a custom accessor ImageAccessor on xarray.DataArray and xarray.Dataset objects.
    This adds methods for 2D manipulation of data and coordinates.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def set_fov(self, fov, image_dims=['y', 'x']):
        """
        Change the field of view of the underlying image data.

        Parameters
        ----------
        fov: float
            Field of view.
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.

        Returns
        -------
        data: xr.DataArray or xr.Dataset
            A DataArray or Dataset with the underlying 2D coordinates updated.

        Notes
        -----
        Assumes a symmetric fov across image dimensions
        """
        data = self._obj
        grid = linspace_2d((data[image_dims[0]].size, data[image_dims[1]].size),
                           (-fov / 2.0, -fov / 2.0), (fov / 2.0, fov / 2.0),
                           endpoint=(True, True))
        data = data.assign_coords({image_dims[0]: grid[image_dims[0]], image_dims[1]: grid[image_dims[1]]})
        data.attrs.update(fov=fov)
        return data

    def get_fov(self, dim='y'):
        """
        Get the field of view from the underlying image data Coordinates.

        Parameters
        ----------
        dim: str, default='x',
            Image data dimension.

        Returns
        -------
        fov: float
            The field of view of the underlying DataArray

        Notes
        -----
        Assumes a symmetric fov across image dimensions
        """
        fov = self._obj[dim][-1] - self._obj[dim][0]
        return fov

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
        grid = pynoisy.utils.linspace_2d(
            num, (self._obj[image_dims[0]][0], self._obj[image_dims[1]][0]),
            (self._obj[image_dims[0]][-1], self._obj[image_dims[1]][-1]))
        return self._obj.interp_like(grid,  method=method)

@xr.register_dataset_accessor("polar")
@xr.register_dataarray_accessor("polar")
class PolarAccessor(object):
    """
    Register a custom accessor PolarAccessor on xarray.DataArray and xarray.Dataset objects.
    This adds methods for polar coordinate processing on a 2D (x,y) grid.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def add_coords(self):
        """
        Add polar coordinates to x,y grid.
        """
        if not('x' in self._obj.coords and 'y' in self._obj.coords):
            raise AttributeError('Coordinates have to contain both x and y')
        x, y = self._obj['x'], self._obj['y']
        yy, xx = np.meshgrid(y, x, indexing='ij')
        r = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
        return self._obj.assign_coords({'r': (['y', 'x'], r), 'theta': (['y', 'x'], theta)})

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
        output = self._obj if 'r' in self._obj.coords else self._obj.polar.add_coords()
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
            self._obj.polar.add_coords()
        r = self._obj['r']
        w = r ** (-1.5)
        if r_cutoff > 0.0:
            w.values[(r < r_cutoff).data] = w.polar.r_cutoff(r_cutoff, r_cutoff ** (-1.5), -1.5 * r_cutoff ** (-2.5),
                                                             0.9 * r_cutoff ** (-1.5)).values[(r < r_cutoff).data]
        w.name = 'w_keplerian'
        w.attrs.update(r_cutoff=r_cutoff)
        return w

@xr.register_dataset_accessor("tensor")
class TensorAccessor(object):
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
        previous_magnitude = self._obj.tensor.v_magnitude
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
            return np.sqrt(self._obj.vx ** 2 + self._obj.vy ** 2)
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
            return np.arctan2(self._obj.vy, self._obj.vx)
        else:
            raise AttributeError('Dataset has to contain both vx and vy')


#########

uniform_sample = lambda a, b: (b - a) * np.random.random_sample() + a

def load_modes(path, dtype=float):
    if dtype == complex:
        modes = read_complex(path)
    elif dtype == float:
        try:
            modes = xr.load_dataarray(path)
        except ValueError:
            modes = xr.load_dataset(path)
    elif dtype == 'eigenvalues':
        modes = xr.open_dataset(path).eigenvalues
    elif dtype == 'eigenvectors':
        modes = xr.open_dataset(path).eigenvectors
    else:
        raise AttributeError('dtype is either float or complex')
    return modes


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

def opening_angles_grf_residuals(files, measurements, envelope=None, damp=1.0, degree=np.inf, return_coefs=False):
    residuals, coefficients = [], []
    for file in tqdm(files):
        modes = load_modes(file, dtype=measurements.dtype)

        residual, coefs = [], []
        degree = min(degree, modes.deg.size)
        for temporal_angle in tqdm(modes.temporal_angle, leave=False):
            modes_reduced = modes.sel(temporal_angle=temporal_angle).isel(deg=slice(degree)).dropna('deg')
            subspace = envelope * modes_reduced.eigenvectors if envelope is not None else modes_reduced.eigenvectors
            subspace *= modes_reduced.eigenvalues
            subspace = subspace.where(np.isfinite(measurements))
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
    residuals.attrs.update(file_num=len(files), damp=damp, with_envelope='False' if envelope is None else 'True')
    residuals.attrs.update(measurements.attrs)
    output = residuals
    if return_coefs:
        coefficients = xr.concat(coefficients, dim='spatial_angle').sortby('spatial_angle')
        coefficients.attrs.update(residuals.attrs)
        output = (residuals, coefficients)

    return output

def find_closest_modes(temporal_angle, spatial_angle, files, dtype=float):
    if isinstance(spatial_angle, str) and (spatial_angle != 'all'):
        raise AttributeError('Invalid attribute, spatial angle should be a float or "all"')
    if isinstance(temporal_angle, str) and (temporal_angle != 'all'):
        raise AttributeError('Invalid attribute, temporal angle should be a float or "all"')

    spatial_angles = xr.concat(
        [xr.open_dataset(file).spatial_angle for file in files], dim='spatial_angle').sortby('spatial_angle')

    if spatial_angle != 'all':
        spatial_angle = spatial_angles.sel(spatial_angle=spatial_angle, method='nearest')

    modes = []
    for file in files:
        if (spatial_angle != 'all') and (xr.open_dataset(file).spatial_angle != spatial_angle):
            continue
        mode = load_modes(file, dtype=dtype)
        if temporal_angle != 'all':
            mode = mode.sel(temporal_angle=temporal_angle, method='nearest')
        modes.append(mode)

    modes = xr.concat(modes, dim='spatial_angle').sortby('spatial_angle').squeeze()
    return modes

def save_complex(dataset, *args, **kwargs):
    ds = dataset.expand_dims('reim', axis=-1) # Add ReIm axis at the end
    ds = xr.concat([ds.real, ds.imag], dim='reim')
    return ds.to_netcdf(*args, **kwargs)

def read_complex(*args, **kwargs):
    ds = xr.open_dataset(*args, **kwargs)
    output = ds.isel(reim=0) + 1j * ds.isel(reim=1)
    output.attrs.update(ds.attrs)
    output = output.to_array().squeeze('variable') if len(output.data_vars.items()) == 1 else output
    return output


def krylov_residual(solver, measurements, degree, n_jobs=4, std_scaling=False):
    error = krylov_error_fn(solver, measurements, degree, n_jobs, std_scaling=std_scaling)
    loss = (error**2).mean()
    return np.array(loss)


def krylov_projection(solver, measurements, degree, n_jobs=4, std_scaling=False):
    krylov = solver.run(source=measurements, nrecur=degree, verbose=0, std_scaling=std_scaling, n_jobs=n_jobs)
    k_matrix = krylov.data.reshape(degree, -1).T
    projection = pynoisy.linalg.least_squares_projection(measurements, k_matrix)
    return projection

def krylov_error_fn(solver, measurements, degree, n_jobs=4, std_scaling=False):
    projection = krylov_projection(solver, measurements, degree, n_jobs, std_scaling=std_scaling)
    error = projection - measurements
    return error



def matern_covariance(nx, ny, length_scale):
    from sklearn.gaussian_process.kernels import Matern
    kernel = Matern(length_scale=length_scale)
    _grid = get_grid(nx, ny)
    x, y = np.meshgrid(_grid.x, _grid.y)
    covariance = kernel(np.array([x.ravel(), y.ravel()]).T)
    return covariance

def slider_select_file(dir, filetype=None):
    from pathlib import Path

    def select_path(i, paths):
        print(paths[i])
        return paths[i]

    filetype = '*' if filetype is None else '*.' + filetype
    paths = [str(path) for path in Path(dir).rglob('{}'.format(filetype))]
    file = interactive(select_path, i=(0, len(paths)-1), paths=fixed(paths));
    display(file)
    return file




@xr.register_dataset_accessor("noisy_methods")
@xr.register_dataarray_accessor("noisy_methods")
class noisy_methods(object):
    def __init__(self, data_array):
        self._obj = data_array

    def to_world_coords(self, tstart, tstop, fov=160.0, xy_units='uas'):

        import ehtim.const_def as ehc

        movies = self._obj
        x = np.linspace(-fov / 2.0, fov / 2.0, movies.x.size)
        y = np.linspace(-fov / 2.0, fov / 2.0, movies.y.size)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        movies = movies.assign_coords({'x': x, 'y': y,
                                       'r': (['x', 'y'], np.sqrt(xx ** 2 + yy ** 2)),
                                       'theta': (['x', 'y'], np.arctan2(yy, xx))})

        psize = (fov * ehc.RADPERUAS) / movies.x.size
        movies.attrs.update(
            fov=fov,
            psize=psize,
            t_units = 'UTC hr',
            xy_units=xy_units
        )

        if 't' in movies.dims:
            movies = movies.assign_coords(t=np.linspace(tstart, tstop, movies.t.size))
            movies.attrs.update(tstart = tstart, tstop = tstop)
        return movies
