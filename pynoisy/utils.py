import noisy_core
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib import animation
from ipywidgets import interact, fixed, interactive
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import display


def matern_covariance(length_scale):
    from sklearn.gaussian_process.kernels import Matern
    kernel = Matern(length_scale=length_scale)
    _grid = get_grid()
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


@xr.register_dataset_accessor("noisy_methods")
@xr.register_dataarray_accessor("noisy_methods")
class noisy_methods(object):
    def __init__(self, data_array):
        self._obj = data_array

    def get_tensor(self):
        tensor = noisy_core.get_diffusion_tensor(
            self._obj.attrs['tensor_ratio'],
            self._obj.spatial_angle.data,
            self._obj.diffusion_coefficient.data
        )
        return tensor

    def plot_principal_axis(self, downscale_factor=8, mode='noisy'):
        """TODO"""
        assert mode in ['noisy', 'hgrf'], "Mode is either noisy or hgrf"
        angle = self._obj.spatial_angle.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(angle.x, angle.y, indexing='xy')
        if mode == 'noisy':
            plt.quiver(x, y, np.sin(angle), np.cos(angle))
        elif mode == 'hgrf':
            plt.quiver(y, x, np.sin(angle), np.cos(angle))
        plt.title('Diffusion tensor (primary)', fontsize=18)

    def plot_secondary_axis(self, downscale_factor=8, mode='noisy'):
        """TODO"""
        assert mode in ['noisy', 'hgrf'], "Mode is either noisy or hgrf"
        angle = self._obj.spatial_angle.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(angle.x, angle.y, indexing='xy')
        if mode == 'noisy':
            plt.quiver(x, y, np.cos(angle), -np.sin(angle))
        elif mode == 'hgrf':
            plt.quiver(y, x, np.cos(angle), -np.sin(angle))
        plt.title('Diffusion tensor (secondary)', fontsize=18)

    def plot_velocity(self, downscale_factor=8, mode='noisy'):
        """TODO"""
        v = self._obj.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(v.x, v.y, indexing='xy')
        if mode == 'noisy':
            plt.quiver(x, y, v.vy, v.vx)
        elif mode == 'hgrf':
            plt.quiver(y, x, v.vx, v.vy)
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
        
    def get_animation(self, vmin=None, vmax=None, fps=10, output=None, cmap='afmhot'):
        """TODO"""
        num_frames, nx, ny = self._obj.sizes.values()
        extent = [self._obj.x.min(), self._obj.x.max(), self._obj.y.min(), self._obj.y.max()]
        # initialization function: plot the background of each frame
        def init():
            im.set_data(np.zeros((nx, ny)), extent=extent, vmin=-1, vmax=1)
            return [im]

        # animation function.  This is called sequentially
        def animate(i):
            im.set_array(self._obj.isel(t=i))
            return [im]

        fig, ax = plt.subplots()
        im = plt.imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap)
        plt.colorbar()
        vmin = self._obj.min() if vmin is None else vmin
        vmax = self._obj.max() if vmax is None else vmax
        im.set_clim(vmin, vmax)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1e3 / fps)

        if output is not None:
            anim.save(output, writer='imagemagick', fps=fps)
        return anim

