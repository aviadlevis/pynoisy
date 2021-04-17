"""
Inverse modeling (inference) classes and functions to compute estimated of SPDE parameters [1] that best match data.
Data could be movie domain data or visibility measurements synthesized using eht-imaging [2].

References
----------
.. [1] Lee, D. and Gammie, C.F., 2021. Disks as Inhomogeneous, Anisotropic Gaussian Random Fields.
    The Astrophysical Journal, 906(1), p.39.
    url: https://iopscience.iop.org/article/10.3847/1538-4357/abc8f3/meta
.. [2] eht-imaging: https://github.com/achael/eht-imaging
"""
import numpy as _np
import xarray as _xr
import gc as _gc
import matplotlib.pyplot as _plt
import pynoisy.linalg as _linalg
import pynoisy.utils as _utils 

@_utils.mode_map('Dataset', ['total', 'data'])
def compute_pixel_loss(modes, measurements, damp=0.0):
    """
    Compute projection residual of the *direct pixel measurements* onto the subspace.
    The function solves ``min ||Ax - b||^2`` or the damped version: ``min ||Ax - b||^2 + d^2 ||x||^2``,
    where A is the subspace matrix, b is the input vector and d is the damping factor.

    Parameters
    ----------
    modes: xr.Dataset,
        A lazy loaded (dask array data) Dataset with the computed eigenvectors and eigenvalues as a function of 'degree'
        and manifold dimensions. To load a dataset from  use: modes = xr.open_mfdataset('directoy/*.nc')
    measurements: xr.DataArray,
        An input DataArray with direct pixel measurements.
    damp: float, default=0.0
        Damping of the least-squares problem. This is a weight on the coefficients l2 norm: damp^2 * ||x||^2

    Returns
    -------
    loss_dataset: xr.Dataset
        A Dataset computed at every manifold grid point with variables:
            'data'=||Ax* - b||^2 ,
            'total'=||Ax* - b||^2 + d^2 ||x*||^2.

    Notes
    -----
    To avoid memory overload subspace is deleted and garbage collected.
    """
    subspace = modes.eigenvalues * modes.eigenvectors
    subspace = subspace.where(_np.isfinite(measurements))
    loss = _linalg.projection_residual(measurements, subspace, damp=damp)

    # Release memory
    del subspace
    _gc.collect()

    return loss

@_xr.register_dataarray_accessor("utils_loss")
class _LossAccessor(object):
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
        minimum_dict: dict.
            An dictionary with the minimum point coordinates.
        """
        data = self._obj.squeeze()
        minimum = data[data.argmin(data.dims)].coords
        minimum_dict = dict((key,float(value)) for key, value in minimum.items())
        return minimum_dict

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
            fig, ax = _plt.subplots(1, 1, figsize=figsize)

        data.plot(ax=ax, color=color)
        dims = data.dims
        if (true_val is not None):
            ax.axvline(true_val, c=vlinecolor, linestyle='--', label='True')
            ax.legend()
        ax.set_ylabel('')
        ax.set_title('Residual Loss', fontsize=fontsize)
        ax.set_xlabel(dims[0])
        ax.set_xlim([float(data[dims[0]].min()), float(data[dims[0]].max())])
        _plt.tight_layout()

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
            fig, ax = _plt.subplots(1, 1, figsize=figsize)

        data.plot(ax=ax, rasterized=rasterized, vmax=vmax, cmap=cmap)
        minimum = data.utils_loss.argmin()
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
                    dim_value = _np.array(dim_value)[_np.argsort(dim_index)]
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
        _plt.tight_layout()