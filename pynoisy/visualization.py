import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation

from ipywidgets import interact, fixed
from mpl_toolkits.axes_grid1 import make_axes_locatable

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



@xr.register_dataarray_accessor("visualization")
class visualization(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def animate(self, t_dim='t', ax=None, vmin=None, vmax=None, cmap=None, add_ticks=True, add_colorbar=True,
                fps=10, output=None):
        """
        Animate a 3D xr.DataArray along a chosen dimension.

        Parameters
        ----------
        t_dim: str, default='t'
            The dimension along which to animate frames
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        vmin, vmax : float, optional
            vmin and vmax define the data range that the colormap covers.
            By default, the colormap covers the complete value range of the supplied data.
        cmap : str or matplotlib.colors.Colormap, optional
            The Colormap instance or registered colormap name used to map scalar data to colors.
            Defaults to :rc:`image.cmap`.
        add_ticks: bool, default=True
            If true then ticks will be visualized.
        add_colorbar: bool, default=True
            If true then a colorbar will be visualized
        fps: float, default=10,
            Frames per seconds.
        output: string,
            Path to save the animated gif. Should end with .gif.

        Returns
        -------
        anim: matplotlib.animation.FuncAnimation
            Animation object.
        """
        movie = self._obj.squeeze()
        if movie.ndim != 3:
            raise AttributeError('Move dimensions ({}) different than 3'.format(movie.ndim))

        num_frames = movie[t_dim].size
        image_dims = list(movie.dims)
        image_dims.remove(t_dim)
        nx, ny = [movie.sizes[dim] for dim in image_dims]

        # Image animation function (called sequentially)
        def animate(i):
            im.set_array(movie.isel({t_dim: i}))
            return [im]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        extent = [movie[image_dims[0]].min(), movie[image_dims[0]].max(),
                  movie[image_dims[1]].min(), movie[image_dims[1]].max()]

        # Initialization function: plot the background of each frame
        im = ax.imshow(np.zeros((nx, ny)), extent=extent, origin='lower', cmap=cmap)
        if add_colorbar:
            fig.colorbar(im)
        if add_ticks == False:
            ax.set_xticks([])
            ax.set_yticks([])
        vmin = movie.min() if vmin is None else vmin
        vmax = movie.max() if vmax is None else vmax
        im.set_clim(vmin, vmax)
        anim = animation.FuncAnimation(fig, animate, frames=num_frames, interval=1e3 / fps)

        if output is not None:
            anim.save(output, writer='imagemagick', fps=fps)
        return anim


    def animate_synced(movie_list, axes, t_dim='t', vmin=None, vmax=None, cmaps=None, add_ticks=False,
                       add_colorbars=True, titles=None, fps=10, output=None):
        """
        Synchronous animation of multiple 3D xr.DataArray along a chosen dimension.

        Parameters
        ----------
        movie_list: list of 3D xr.DataArrays
            A list of movies to animated synchroniously.
        axes: list of matplotlib axis,
            List of matplotlib axis object for the visualization. Should have same length as movie_list.
        t_dim: str, default='t'
            The dimension along which to animate frames
        vmin, vmax : float, optional
            vmin and vmax define the data range that the colormap covers.
            By default, the colormap covers the complete value range of the supplied data.
        cmaps : list of str or matplotlib.colors.Colormap, optional
            If this is a scalar then it is extended for all movies.
            The Colormap instance or registered colormap name used to map scalar data to colors.
            Defaults to :rc:`image.cmap`.
        add_ticks: bool, default=True
            If true then ticks will be visualized.
        add_colorbars: list of bool, default=True
            If this is a scalar then it is extended for all movies. If true then a colorbar will be visualized.
        titles: list of strings, optional
            List of titles for each animation. Should have same length as movie_list.
        fps: float, default=10,
            Frames per seconds.
        output: string,
            Path to save the animated gif. Should end with .gif.

        Returns
        -------
        anim: matplotlib.animation.FuncAnimation
            Animation object.
        """
        # Image animation function (called sequentially)
        def animate(i):
            for movie, im in zip(movie_list, images):
                im.set_array(movie.isel({t_dim: i}))
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
            if add_ticks == False:
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

@xr.register_dataset_accessor("visualization")
class visualization(object):
    def __init__(self, data_array):
        self._obj = data_array

    def major_axis(self, ax=None, figsize=(5,4), downscale_factor=8, color='black', alpha=1.0, width=None, scale=None):
        """
        Quiver plot of spatial correlation major axis.

        Parameters
        ----------
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        figsize: (float, float),
            Figure size: (horizontal_size, vertical_size)
        downscale_factor: int, default=8
            Downscale the quiver image. Note that this could cause interpolation issues for high downscales or
            factors that are not powers of two.
        color : color or color sequence, optional
            Explicit color(s) for the arrows. If *C* has been set, *color* has no effect.
        alpha: float (0, 1.0)
            Alpha transparency for the quiver.
        width: float, optional
            Width of the arrows.
        scale : float, optional
            Number of data units per arrow length unit, e.g., m/s per plot width; a
            smaller scale parameter makes the arrow longer.
            If *None*, a simple autoscaling algorithm is used, based on the average
            vector length and the number of vectors. The arrow length unit is given by
            the *scale_units* parameter.
        """
        if not 'spatial_angle' in self._obj:
            raise AttributeError('Dataset doesnt have spatial_angle DataArray')
        spatial_angle = self._obs.spatial_angle
        if not ('x' in spatial_angle.coords and 'y' in spatial_angle.coords):
            raise AttributeError('spatial_angle doesnt have both x and y coordinates')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        angle = spatial_angle.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(angle.x, angle.y, indexing='xy')
        ax.quiver(y, x, np.sin(angle), np.cos(angle), headaxislength=0, headlength=0, color=color,
                  alpha=alpha, width=width, scale=scale)
        ax.set_title('Spatial correlation major axis')

    def minor_axis(self, ax=None, figsize=(5,4), downscale_factor=8, color='black', alpha=1.0, width=None, scale=None):
        """
        Quiver plot of spatial correlation minor axis.

        Parameters
        ----------
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        figsize: (float, float),
            Figure size: (horizontal_size, vertical_size)
        downscale_factor: int, default=8
            Downscale the quiver image. Note that this could cause interpolation issues for high downscales or
            factors that are not powers of two.
        color : color or color sequence, optional
            Explicit color(s) for the arrows. If *C* has been set, *color* has no effect.
        alpha: float (0, 1.0)
            Alpha transparency for the quiver.
        width: float, optional
            Width of the arrows.
        scale : float, optional
            Number of data units per arrow length unit, e.g., m/s per plot width; a
            smaller scale parameter makes the arrow longer.
            If *None*, a simple autoscaling algorithm is used, based on the average
            vector length and the number of vectors. The arrow length unit is given by
            the *scale_units* parameter.
        """
        if not 'spatial_angle' in self._obj:
            raise AttributeError('Dataset doesnt have spatial_angle DataArray')
        spatial_angle = self._obs.spatial_angle
        if not ('x' in spatial_angle.coords and 'y' in spatial_angle.coords):
            raise AttributeError('spatial_angle doesnt have both x and y coordinates')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        angle = spatial_angle.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(angle.x, angle.y, indexing='xy')
        ax.quiver(y, x, np.cos(angle), -np.sin(angle), headaxislength=0, headlength=0, color=color, alpha=alpha,
                   width=width, scale=scale)
        ax.set_title('Spatial correlation minor axis')

    def velocity_quiver(self, ax=None, figsize=(5,4), downscale_factor=8, color='black', alpha=1.0, width=None, scale=None):
        """
        Quiver plot of velocity field.

        Parameters
        ----------
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        figsize: (float, float),
            Figure size: (horizontal_size, vertical_size)
        downscale_factor: int, default=8
            Downscale the quiver image. Note that this could cause interpolation issues for high downscales or
            factors that are not powers of two.
        color : color or color sequence, optional
            Explicit color(s) for the arrows. If *C* has been set, *color* has no effect.
        alpha: float (0, 1.0)
            Alpha transparency for the quiver.
        width: float, optional
            Width of the arrows.
        scale : float, optional
            Number of data units per arrow length unit, e.g., m/s per plot width; a
            smaller scale parameter makes the arrow longer.
            If *None*, a simple autoscaling algorithm is used, based on the average
            vector length and the number of vectors. The arrow length unit is given by
            the *scale_units* parameter.
        """
        if not ('vx' in self._obj and 'vy' in self._obj):
            raise AttributeError('Dataset doesnt have both vx and vy')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        v = self._obj.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        x, y = np.meshgrid(v.x, v.y, indexing='xy')
        ax.quiver(y, x, v.vx, v.vy, color=color, width=width, scale=scale, alpha=alpha)
        ax.set_title('Velocity field')
