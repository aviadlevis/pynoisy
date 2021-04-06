"""
Methods and function useful for visualization of Gaussian Random Fields (GRFs) and the diffusion tensor parameters:
velocity field, spatial correlation axis etc. This includes various animation methods and interactive sliders for easy
visualization of spatio-temporal fields and quiver plots for directional data fields.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import animation
from ipywidgets import interact
from mpl_toolkits.axes_grid1 import make_axes_locatable

def slider_frame_comparison(frames1, frames2, scale='amp'):
    """
    Slider comparison of two 3D xr.DataArray along a chosen dimension.

    Parameters
    ----------
    frames1: xr.DataArray
        A 3D DataArray with 't' dimension to compare along
    frames2:  xr.DataArray
        A 3D DataArray with 't' dimension to compare along
    scale: 'amp' or 'log', default='amp'
        Compare absolute values or log of the fractional deviation.
    """
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

    def imshow_frame(frame):
        image1 = frames1[frame]
        image2 = frames2[frame]

        if scale == 'amp':
            image3 = np.abs(frames1[frame] - frames2[frame])
        elif scale == 'log':
            image3 = np.log(np.abs(frames1[frame]/frames2[frame]))

        for ax, img, title, cbar in zip(axes, [image1, image2, image3], titles, cbars):
            ax.imshow(img, origin='lower')
            ax.set_title(title)
            cbar.mappable.set_clim([img.min(), img.max()])

    num_frames = min(frames1.t.size, frames2.t.size)
    plt.tight_layout()
    interact(imshow_frame, frame=(0, num_frames-1));

def animate_synced(movie_list, axes, t_dim='t', vmin=None, vmax=None, cmaps='RdBu_r', add_ticks=False,
                   add_colorbars=True, titles=None, fps=10, output=None):
    """
    Synchronous animation of multiple 3D xr.DataArray along a chosen dimension.

    Parameters
    ----------
    movie_list: list of xr.DataArrays
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
    vmin_list = [movie.min() for movie in movie_list] if vmin is None else vmin
    vmax_list = [movie.max() for movie in movie_list] if vmax is None else vmax

    for movie, ax, title, cmap, vmin, vmax in zip(movie_list, axes, titles, cmaps, vmin_list, vmax_list):
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

def plot_mean_std(mean, std, x=None, ax=None, color='tab:blue', alpha=0.35):
    """
    Plot mean and semi-transparent standard deviation.

    Parameters
    ----------
    mean: list or np.array,
        A list or numpy array with mean values. len(mean)=len(std).
    std: list or np.array,
        A list or numpy array with standard deviation values. len(std)=len(mean).
    x: list or np.array, optional,
        Horizontal axis. If None, a simple range is used.
    ax: matplotlib axis,
        A matplotlib axis object for the visualization.
    true_color: color, default='tab:blue'
        Color of the plot.
    alpha: float in range (0, 1), default=0.35
        Alpha transparency for the standard deviation.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    if len(mean) != len(std):
        raise AttributeError('mean length ({}) != std length ({})'.format(len(mean), len(std)))
    mean = np.array(mean)
    std = np.array(std)
    x = range(len(mean)) if x is None else x
    ax.plot(x, mean, color=color)
    ax.fill_between(x, mean + std, mean - std, facecolor=color, alpha=alpha)
    ax.set_xlim([x[0], x[-1]])


@xr.register_dataarray_accessor("visualization")
class VisualizationAccessor(object):
    """
    Register a custom accessor VisualizationAccessor on xarray.DataArray object.
    This adds methods for visualization of Gaussian Random Fields (3D DataArrays) along a single axis.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def slider(self, t_dim='t', ax=None, cmap=None):
        """
        Interactive slider visualization of a 3D xr.DataArray along specified dimension.

        Parameters
        ----------
        t_dim: str, default='t'
            The dimension along which to animate frames
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        cmap : str or matplotlib.colors.Colormap, optional
            The Colormap instance or registered colormap name used to map scalar data to colors.
            Defaults to :rc:`image.cmap`.
        """
        movie = self._obj.squeeze()
        if movie.ndim != 3:
            raise AttributeError('Move dimensions ({}) different than 3'.format(movie.ndim))

        num_frames = movie[t_dim].size
        image_dims = list(movie.dims)
        image_dims.remove(t_dim)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        extent = [movie[image_dims[0]].min(), movie[image_dims[0]].max(),
                  movie[image_dims[1]].min(), movie[image_dims[1]].max()]

        im = ax.imshow(movie.isel({t_dim: 0}), extent=extent, cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        def imshow_frame(frame):
            img = movie.isel({t_dim: frame})
            ax.imshow(img, origin='lower', extent=extent, cmap=cmap)
            cbar.mappable.set_clim([img.min(), img.max()])

        interact(imshow_frame, frame=(0, num_frames-1));

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

@xr.register_dataarray_accessor("loss")
class LossAccessor(object):
    """
    Register a custom accessor LossAccessor on xarray.DataArray object.
    This adds methods for processing and visualization of loss manifolds.
    """
    def __init__(self, data_array):
        self._obj = data_array

    def argmin(self):
        """
        Find the minimum point coordinates

        Returns
        -------
        minimum: xr.Coordinates.
            An xarray Coordinate object with the minimum point coordinates.
        """
        data = self._obj.squeeze()
        minimum = data[data.argmin(data.dims)].coords
        return minimum

    def plot1d(self, true_val=None, ax=None, figsize=(5,4), color=None, vlinecolor='red', fontsize=16):
        """
        1D plot of loss curve.

        Parameters
        ----------
        true_val: float, optional
            The true underlying parameter value. Setting this value will plot a vertical line.
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        figsize: (float, float),
            Figure size: (horizontal_size, vertical_size)
        color: color, optional
            Color of the plot data points.
        vlinecolor: color, default='red',
            Color of the vertical line marking the true underlying parameters. Only affects if true_val is set.
        fontsize: float, default=16,
            fontsize of the title.
        """
        data = self._obj.squeeze()

        if (data.ndim != 1):
            raise AttributeError('Loss curve should have dimension 1.')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        data.plot(ax=ax, color=color)
        dims = data.dims
        if (true_val is not None):
            ax.axvline(true_val, c=vlinecolor, linestyle='--', label='True')
            ax.legend()
        ax.set_ylabel('')
        ax.set_title('Residual Loss', fontsize=fontsize)
        ax.set_xlabel(dims[0])
        ax.set_xlim([float(data[dims[0]].min()), float(data[dims[0]].max())])
        plt.tight_layout()

    def plot2d(self, true=None, ax=None, figsize=(5,4), contours=False, rasterized=False, vmax=None,
               cmap=None, fontsize=16, linewidth=2.5, s=100, true_color='w', minimum_color='r'):
        """
        2D plot of loss manifold.

        Parameters
        ----------
        true: dictionary, optional
            A dictionary with the true underlying parameters specified values.
        ax: matplotlib axis,
            A matplotlib axis object for the visualization.
        figsize: (float, float),
            Figure size: (horizontal_size, vertical_size)
        contours: bool, default=False
            Plot contours on top of the 2D manifold.
        rasterized: bool, default=False,
            Set true for saving vector graphics.
        vmax : float, optional
            vmax defines the data range maximum that the colormap covers.
            By default, the colormap covers the complete value range of the supplied data.
        cmap : str or matplotlib.colors.Colormap, optional
            The Colormap instance or registered colormap name used to map scalar data to colors.
            Defaults to :rc:`image.cmap`.
        fontsize: float, default=16,
            fontsize of the title.
        linewidth: float, default=2.5,
            Linewidth of the True underlying parameters. Only effects if the true dictionary has *one* element.
        s: float, default=100,
            Size of scatter plot data points (minimum and true)
        true_color: color, default='white'
            Color of the true data point scatter plot. Only effects if true dictionary has *two* elements.
        minimum_color: color, default='red',
            Color of the minimum point scatter plot.
        """
        data = self._obj.squeeze()

        if (data.ndim != 2):
            raise AttributeError('Loss manifold has dimension different than 2')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        data.plot(ax=ax, rasterized=rasterized, vmax=vmax, cmap=cmap)
        minimum = data.loss.argmin()
        dims = data.dims

        ax.scatter(minimum[dims[1]], minimum[dims[0]], s=s, c=minimum_color, marker='o', label='Global minimum')
        if (true is not None):
            if isinstance(true, dict):
                dim_index, dim_value = [], []
                for key, val in true.items():
                    dim_index.append(list(dims).index(key))
                    dim_value.append(val)

                # Plot a horizontal or vertical line for the true value
                if (len(dim_value) == 1):
                    if (dim_index[0] == 0):
                        ax.axhline(dim_value[0], c=true_color, linestyle='--', label='True', linewidth=linewidth)
                    else:
                        ax.axvline(dim_value[0], c=true_color, linestyle='--', label='True', linewidth=linewidth)

                # Plot a point for the true value
                elif (len(dim_value) == 2):
                    dim_value = np.array(dim_value)[np.argsort(dim_index)]
                    ax.scatter(dim_value[1], dim_value[0], s=s, c=true_color, marker='^', label='True')

                else:
                    raise AttributeError('True dictionary has axis dimensions than the data')
            else:
                raise AttributeError('True should be a dictionary of coordinates and values.')

        if contours:
            cs = data.plot.contour(ax=ax, cmap='RdBu_r')
            ax.clabel(cs, inline=1, fontsize=10)

        ax.legend(facecolor='white', framealpha=0.4)
        ax.set_title('Residual Loss', fontsize=fontsize)
        plt.tight_layout()

@xr.register_dataset_accessor("visualization")
class VisualizationAccessor(object):
    """
    Register a custom accessor VisualizationAccessor on xarray.Dataset object.
    This adds methods for visualization of diffusion tensor elements.
    """
    def __init__(self, data_array):
        self._obj = data_array

    def major_axis(self, ax=None, figsize=(5,4), downscale_factor=8, color='black', alpha=1.0, width=None, scale=None,
                   fontsize=12):
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
        alpha: float in range (0, 1)
            Alpha transparency for the quiver.
        width: float, optional
            Width of the arrows.
        scale : float, optional
            Number of data units per arrow length unit, e.g., m/s per plot width; a
            smaller scale parameter makes the arrow longer.
            If *None*, a simple autoscaling algorithm is used, based on the average
            vector length and the number of vectors. The arrow length unit is given by
            the *scale_units* parameter.
        fontsize: float, default=12,
            Title fontsize
        """
        if not 'spatial_angle' in self._obj:
            raise AttributeError('Dataset doesnt have spatial_angle DataArray')
        spatial_angle = self._obj.spatial_angle
        if not ('x' in spatial_angle.coords and 'y' in spatial_angle.coords):
            raise AttributeError('spatial_angle doesnt have both x and y coordinates')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        angle = spatial_angle.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        xx, yy = np.meshgrid(angle.x, angle.y, indexing='xy')
        ax.quiver(xx, yy, np.cos(angle), np.sin(angle), headaxislength=0, headlength=0, color=color,
                  alpha=alpha, width=width, scale=scale)
        ax.set_title('Spatial correlation major axis', fontsize=fontsize)
        ax.set_aspect('equal')

    def minor_axis(self, ax=None, figsize=(5,4), downscale_factor=8, color='black', alpha=1.0, width=None, scale=None,
                   fontsize=12):
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
        fontsize: float, default=12,
            Title fontsize
        """
        if not 'spatial_angle' in self._obj:
            raise AttributeError('Dataset doesnt have spatial_angle DataArray')
        spatial_angle = self._obj.spatial_angle
        if not ('x' in spatial_angle.coords and 'y' in spatial_angle.coords):
            raise AttributeError('spatial_angle doesnt have both x and y coordinates')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        angle = spatial_angle.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        xx, yy = np.meshgrid(angle.x, angle.y, indexing='xy')
        ax.quiver(xx, yy, -np.sin(angle), np.cos(angle), headaxislength=0, headlength=0, color=color, alpha=alpha,
                   width=width, scale=scale)
        ax.set_title('Spatial correlation minor axis', fontsize=fontsize)
        ax.set_aspect('equal')

    def velocity_quiver(self, ax=None, figsize=(5,4), downscale_factor=8, color='black', alpha=1.0, width=None,
                        scale=None, fontsize=12):
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
        fontsize: float, default=12,
            Title fontsize
        """
        if not ('vx' in self._obj and 'vy' in self._obj):
            raise AttributeError('Dataset doesnt have both vx and vy')

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        v = self._obj.coarsen(x=downscale_factor, y=downscale_factor, boundary='trim').mean()
        xx, yy = np.meshgrid(v.x, v.y, indexing='xy')
        ax.quiver(xx, yy, v.vx, v.vy, color=color, width=width, scale=scale, alpha=alpha)
        ax.set_title('Velocity field', fontsize=fontsize)
        ax.set_aspect('equal')
