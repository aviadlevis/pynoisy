import noisy_core
import pynoisy.algebra_utils
from pynoisy import eht_functions as ehtf
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib import animation
from ipywidgets import interact, fixed, interactive
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import display
import h5py
import os
from tqdm.auto import tqdm
import gc

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

def visualization_2d(residuals, ax=None, degree=None, contours=False, rasterized=False, vmax=None, cmap=None):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(5, 4))
    dataset = residuals.sel(deg=degree) if degree else residuals
    minimum = dataset[dataset.argmin(dim=['temporal_angle', 'spatial_angle'])].coords
    dataset.plot(ax=ax, add_labels=False, rasterized=rasterized, vmax=vmax, cmap=cmap)
    ax.scatter(minimum['temporal_angle'], minimum['spatial_angle'], s=100, c='r', marker='o', label='Global minimum')
    if hasattr(residuals, 'true_temporal_angle'):
        ax.scatter(residuals.true_temporal_angle, residuals.true_spatial_angle, s=100, c='w', marker='^', label='True')
    if contours:
        cs = dataset.plot.contour(ax=ax, cmap='RdBu_r')
        ax.clabel(cs, inline=1, fontsize=10)
    ax.set_title('Residual Loss (degree={})'.format(int(dataset.deg.data)),fontsize=16)
    ax.set_xlabel('Temporal angle [rad]', fontsize=12)
    ax.set_ylabel('Spatial angle [rad]', fontsize=12)
    ax.legend(facecolor='white', framealpha=0.4)


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

            output = pynoisy.algebra_utils.projection_residual(measurements, subspace, damp=damp, return_coefs=return_coefs)

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
            output = pynoisy.algebra_utils.projection_residual(measurements, subspace, damp=damp, return_coefs=return_coefs)
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
    projection = pynoisy.algebra_utils.least_squares_projection(measurements, k_matrix)
    return projection

def krylov_error_fn(solver, measurements, degree, n_jobs=4, std_scaling=False):
    projection = krylov_projection(solver, measurements, degree, n_jobs, std_scaling=std_scaling)
    error = projection - measurements
    return error

def full_like(nx, ny, fill_value):
    _grid = get_grid(nx, ny)
    array = xr.full_like(xr.DataArray(coords=_grid.coords), fill_value=fill_value)
    return array

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

def get_grid(nx, ny):
    """TODO"""
    x, y = noisy_core.get_xy_grid(nx, ny)
    grid = xr.Dataset(
        coords={'x': x[:, 0], 'y': y[0],
                'r': (['x', 'y'], np.sqrt(x ** 2 + y ** 2)),
                'theta': (['x', 'y'], np.arctan2(y, x))
                }
    )
    return grid

def compare_movie_frames(frames1, frames2, scale='amp'):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plt.tight_layout()
    mean_images = [frames1.mean(axis=0), frames2.mean(axis=0),
                   (np.abs(frames1 - frames2)).mean(axis=0)]
    cbars = []
    titles = [None]*3
    titles[0] = frames1.name if frames1.name is not None else 'Movie1'
    titles[1] = frames2.name if frames2.name is not None else 'Movie2'
    if scale == 'amp':
        titles[2] = 'Absolute difference'
    elif scale == 'log':
        titles[2] = 'Log relative difference'

    for ax, image in zip(axes, mean_images):
        im = ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbars.append(fig.colorbar(im, cax=cax))

    def imshow_frame(i, frames1, frames2, axes, cbars):
        image1 = frames1[i]
        image2 = frames2[i]

        if scale == 'amp':
            image3 = np.abs(frames1[i] - frames2[i])
        elif scale == 'log':
            image3 = np.log(np.abs(frames1[i]/frames2[i]))

        for ax, img, title, cbar in zip(axes, [image1, image2, image3], titles, cbars):
            ax.imshow(img, origin='lower')
            ax.set_title(title)
            cbar.mappable.set_clim([img.min(), img.max()])

    num_frames = min(frames1.t.size, frames2.t.size)
    interact(
        imshow_frame, i=(0, num_frames - 1),
        frames1=fixed(frames1), frames2=fixed(frames2), axes=fixed(axes), cbars=fixed(cbars)
    );

def get_krylov_matrix(b, forward_fn, degree):
    """
    Compute a matrix with column vectors spanning the Krylov subspace of a certain degree.
    This is done by senquential applications of the forward operator.

    Parameters
    ----------
    b: xr.DataArray or numpy.ndarray, shape=(num_frames, nx, ny)
        b is a video, typically the measurement vector.
    forward_fn: function,
        The forward function, should be self-adjoint or support senquential applications: F(F(b))
    degree: int
        The degree of the Krylov matrix.

    Returns
    -------
    k_matrix: np.ndarray(shape=(degree, b.size))
        The Krylov matrix with column vectors: (F(b), F(F(b)), F(F(F(b)))...)
    """
    k_matrix = []
    for i in range(degree):
        b = forward_fn(b)
        k_matrix.append(np.array(b).ravel())
    return np.array(k_matrix)

def load_grmhd(filepath):
    """
    Load an .h5 GRMHD movie frames

    Parameters
    ----------
    filepath: str
        path to .h5 file

    Returns
    -------
    measurements: xr.DataArray(dims=['t', 'x', 'y'])
        GRMHD measurements as an xarray object.
    """
    filename =  os.path.abspath(filepath).split('/')[-1][:-3]
    with h5py.File(filepath, 'r') as file:
        frames = file['I'][:]
    nt_, nx_, ny_ = frames.shape
    grid = get_grid(nx_, ny_)
    measurements = xr.DataArray(data=frames,
                                coords={'x': grid.x,  'y': grid.y, 't': np.linspace(0, 0.1, nt_)},
                                dims=['t', 'x', 'y'],
                                attrs={'GRMHD': filename})
    return measurements

def grmhd_preprocessing(movie, flux_threshold=1e-10):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        measurements = np.log(movie.where(movie > flux_threshold))
    measurements = measurements - measurements.mean('t')
    return measurements

def multiple_animations(movie_list, axes, vmin=None, vmax=None, titles=None, ticks=False, add_colorbars=True, fps=10, output=None, cmaps='viridis'):
    """TODO"""
    # animation function.  This is called sequentially
    def animate(i):
        for movie, im in zip(movie_list, images):
            im.set_array(movie.isel(t=i))
        return images

    fig = plt.gcf()
    num_frames, nx, ny = movie_list[0].sizes.values()

    image_dims = list(movie_list[0].sizes.keys())
    image_dims.remove('t')
    extent = [movie_list[0][image_dims[0]].min(), movie_list[0][image_dims[0]].max(),
              movie_list[0][image_dims[1]].min(), movie_list[0][image_dims[1]].max()]

    # initialization function: plot the background of each frame
    images = []
    titles = [movie.name for movie in movie_list] if titles is None else titles
    cmaps = [cmaps]*len(movie_list) if isinstance(cmaps, str) else cmaps
    vmin = [movie.min() for movie in movie_list] if vmin is None else vmin
    vmax = [movie.max() for movie in movie_list] if vmax is None else vmax

    for movie, ax, title, cmap, vmin, vmax in zip(movie_list, axes, titles, cmaps, vmin, vmax):
        if ticks == False:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap)
        if add_colorbars:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax)
        im.set_clim(vmin, vmax)
        images.append(im)

    plt.tight_layout()
    anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1e3 / fps)

    if output is not None:
        anim.save(output, writer='imagemagick', fps=fps)
    return anim

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

    def get_projection_matrix(self, constant=1.0):
        modes = self._obj.squeeze()
        if modes.dtype == complex:
            if modes.ndim == 2:
                dims = ('index', 'deg')
            elif modes.ndim == 4:
                dims = ('t', 'u', 'v', 'deg')
            else:
                raise AttributeError('ndim = {} not supported for complex dtype'.format(modes.ndim))
        else:
            dims = ('t', 'x', 'y', 'deg')

        potential_squeeze_dims = list(modes.dims)
        if 'deg' in potential_squeeze_dims:
            potential_squeeze_dims.remove('deg')
        squeeze_dims = []
        for dim in potential_squeeze_dims:
            if modes.sizes[dim] == 1:
                squeeze_dims.append(dim)
        modes = modes.squeeze(squeeze_dims)
        matrix = modes.transpose(*dims).data.reshape(-1, modes.deg.size)

        return matrix

    def get_tensor(self):
        tensor = noisy_core.get_diffusion_tensor(
            self._obj.attrs['tensor_ratio'],
            self._obj.spatial_angle.data,
            self._obj.diffusion_coefficient.data
        )
        return tensor

    def plot_principal_axis(self, downscale_factor=8, mode='noisy', color='black', alpha=1.0, width=None, scale=None):
        """TODO"""
        assert mode in ['noisy', 'hgrf'], "Mode is either noisy or hgrf"
        angle = self._obj.spatial_angle.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(angle.x, angle.y, indexing='xy')
        if mode == 'noisy':
            plt.quiver(x, y, np.sin(angle), np.cos(angle), headaxislength=0, headlength=0, color=color, alpha=alpha, width=width, scale=scale)
        elif mode == 'hgrf':
            plt.quiver(y, x, np.sin(angle), np.cos(angle), headaxislength=0, headlength=0, color=color, alpha=alpha, width=width, scale=scale)
        plt.title('Diffusion tensor (primary)', fontsize=18)

    def plot_secondary_axis(self, downscale_factor=8, mode='noisy', color='black', alpha=1.0, width=None, scale=None):
        """TODO"""
        assert mode in ['noisy', 'hgrf'], "Mode is either noisy or hgrf"
        angle = self._obj.spatial_angle.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(angle.x, angle.y, indexing='xy')
        if mode == 'noisy':
            plt.quiver(x, y, np.cos(angle), -np.sin(angle), headaxislength=0, headlength=0, color=color, alpha=alpha, width=width, scale=scale)
        elif mode == 'hgrf':
            plt.quiver(y, x, np.cos(angle), -np.sin(angle), headaxislength=0, headlength=0, color=color, alpha=alpha, width=width, scale=scale)
        plt.title('Diffusion tensor (secondary)', fontsize=18)

    def plot_velocity(self, downscale_factor=8, mode='noisy', color='black', width=None, scale=None):
        """TODO"""
        v = self._obj.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(v.x, v.y, indexing='xy')
        if mode == 'noisy':
            plt.quiver(x, y, v.vy, v.vx, color=color, width=width, scale=scale)
        elif mode == 'hgrf':
            plt.quiver(y, x, v.vx, v.vy, color=color, width=width, scale=scale)
        plt.title('Velocity field', fontsize=18)

    def plot_statistics(self):
        """TODO"""
        fig, ax = plt.subplots()
        ax.set_title('{} samples statistics'.format(self._obj.num_samples), fontsize=16)
        x_range = self._obj.coords[list(self._obj.coords.keys())[0]]
        mean = self._obj['mean']
        std = self._obj['std']
        ax.plot(x_range, mean)
        ax.fill_between(x_range, mean + std, mean - std, facecolor='blue', alpha=0.5)
        ax.set_xlim([x_range.min(), x_range.max()])
        
    def get_animation(self, vmin=None, vmax=None, fps=10, output=None, cmap='afmhot', ax=None, add_colorbar=True):
        """TODO"""
        # animation function.  This is called sequentially
        def animate(i):
            im.set_array(self._obj.isel(t=i))
            return [im]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        num_frames, nx, ny = self._obj.sizes.values()
        image_dims = list(self._obj.sizes.keys())
        image_dims.remove('t')

        extent = [self._obj[image_dims[0]].min(), self._obj[image_dims[0]].max(),
                  self._obj[image_dims[1]].min(), self._obj[image_dims[1]].max()]

        # initialization function: plot the background of each frame
        im = ax.imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap)
        if add_colorbar:
            fig.colorbar(im)
        vmin = self._obj.min() if vmin is None else vmin
        vmax = self._obj.max() if vmax is None else vmax
        im.set_clim(vmin, vmax)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1e3 / fps)

        if output is not None:
            anim.save(output, writer='imagemagick', fps=fps)
        return anim

    def update_angle_and_magnitude(self):
        """TODO"""
        angle = np.arctan2(self._obj.vy, self._obj.vx)
        magnitude = np.sqrt(self._obj.vy ** 2 + self._obj.vx ** 2)
        self._obj.update({'angle': angle, 'magnitude': magnitude})

    def update_vx_vy(self):
        """TODO"""
        self._obj.vx[:] = np.cos(self._obj.angle) * self._obj.magnitude
        self._obj.vy[:] = np.sin(self._obj.angle) * self._obj.magnitude