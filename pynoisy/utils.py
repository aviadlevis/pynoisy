import core
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from matplotlib import animation
from ipywidgets import interact, fixed
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_grid():
    """TODO"""
    x, y = core.get_xy_grid()
    grid = xr.Dataset(
        coords={'x': x[:, 0], 'y': y[0],
                'r': (['x', 'y'], np.sqrt(x ** 2 + y ** 2)),
                'theta': (['x', 'y'], np.arctan2(y, x))
                }
    )
    return grid

def compare_movie_frames(frames1, frames2):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    plt.tight_layout()
    mean_images = [frames1.mean(axis=0), frames2.mean(axis=0),
                   (np.abs(frames1 - frames2)).mean(axis=0)]
    cbars = []
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
        image3 = np.abs(frames1[i] - frames2[i])

        for ax, img, title, cbar in zip(axes, [image1, image2, image3],
                                        ['Movie1', 'Movie2', 'Absolute difference'], cbars):
            im = ax.imshow(img)
            ax.set_title(title)
            cbar.mappable.set_clim([img.min(), img.max()])

    num_frames = min(frames1.sizes['t'], frames2.sizes['t'])
    interact(
        imshow_frame, i=(0, num_frames - 1),
        frames1=fixed(frames1), frames2=fixed(frames2), axes=fixed(axes), cbars=fixed(cbars)
    );


@xr.register_dataset_accessor("noisy_methods")
@xr.register_dataarray_accessor("noisy_methods")
class noisy_methods(object):
    def __init__(self, data_array):
        self._obj = data_array

    def get_tensor(self):
        tensor = core.get_diffusion_tensor(
            self._obj.attrs['tensor_ratio'],
            self._obj.principle_angle.data,
            self._obj.diffusion_coefficient.data
        )
        return tensor

    def get_laplacian(self, frames, advection):
        laplacian = core.get_laplacian(
            self._obj.attrs['tensor_ratio'],
            self._obj.principle_angle.data,
            self._obj.diffusion_coefficient.data,
            np.array(advection.data, dtype=np.float64, order='C'),
            self._obj.correlation_time.data,
            np.array(frames, dtype=np.float64, order='C')
        )
        return laplacian

    def plot_principal_axis(self, downscale_factor=8):
        """TODO"""
        angle = self._obj.principle_angle.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(angle.x, angle.y, indexing='ij')
        plt.quiver(x, y, np.cos(angle), np.sin(angle))
        plt.title('Diffusion tensor (primary)', fontsize=18)

    def plot_secondary_axis(self, downscale_factor=8):
        """TODO"""
        angle = self._obj.principle_angle.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(angle.x, angle.y, indexing='ij')
        plt.quiver(x, y, -np.sin(angle), np.cos(angle))
        plt.title('Diffusion tensor (secondary)', fontsize=18)

    def plot_velocity(self, downscale_factor=8):
        """TODO"""
        v = self._obj.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(v.x, v.y, indexing='ij')
        plt.quiver(x, y, v.vx, v.vy)
        plt.title('Velocity field', fontsize=18)

    def get_animation(self, vmin=None, vmax=None, fps=10, output=None):
        """TODO"""
        num_frames, nx, ny = self._obj.sizes.values()

        # initialization function: plot the background of each frame
        def init():
            im.set_data(np.zeros((nx, ny)), vmin=-5, vmax=5)
            return [im]

        # animation function.  This is called sequentially
        def animate(i):
            im.set_array(self._obj.isel(t=i))
            return [im]

        fig, ax = plt.subplots()
        im = plt.imshow(np.zeros((nx, ny)))
        plt.colorbar()
        vmin = self._obj.min() if vmin is None else vmin
        vmax = self._obj.min() if vmax is None else vmax
        im.set_clim(vmin, vmax)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1e3 / fps)

        if output is not None:
            anim.save(output, writer='imagemagick', fps=fps)
        return anim

