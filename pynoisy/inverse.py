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
import numpy as np
import xarray as xr
import pynoisy.linalg
import gc

def compute_loss_manifold(loss_fn, modes, measurements, progress_bar=True, kwargs={}):
    """
    Compute loss manifold using using dask parallel processing.

    Parameters
    ----------
    loss_fn:  pynoisy.inverse.dask_loss,
        A decorated loss function (@dask_loss) with inputs: loss_fn(subspace, measurements, **kwargs).
    modes: xr.Dataset,
        A lazy loaded (dask array data) Dataset with the computed eigenvectors and eigenvalues as a function of 'degree'
        and manifold dimensions. To load a dataset from  use: modes = xr.open_mfdataset('directoy/*.nc')
    measurements: xr.DataArray,
        Measurement DataArray. This could be e.g. movie pixels or visibility measurements.
    progress_bar: bool, default=True,
        Progress bar is useful as manifold computations can be time consuming.
    kwargs: dictionary, optional,
        Keyword arguments for the loss_fun.

    Returns
    -------
    loss_dataset: xr.Dataset
        A Dataset computed at every manifold grid point with variables:
            'data'=||Ax* - b||^2 ,
            'total'=||Ax* - b||^2 + d^2 ||x*||^2.
    """

    # Manifold dimensions are the modes dimensions without ['degree', 't', 'x', 'y'].
    coords = modes.coords.to_dataset()
    for dim in ['degree', 't', 'x', 'y']:
        del coords[dim]
    dim_names = list(coords.dims.keys())
    dim_sizes = list(coords.dims.values())

    # Generate an output template for dask which fits in a single chunk.
    template = xr.Dataset(
        coords=coords, data_vars={'data': (dim_names, np.empty(dim_sizes)),
                                  'total': (dim_names, np.empty(dim_sizes))}
    ).chunk(dict(zip(dim_names, [1] * len(dim_names))))

    # Generate dask computation graph
    mapped = xr.map_blocks(loss_fn, modes, args=(measurements, dim_names), kwargs=kwargs, template=template)

    # Preform actual computation with or without progress bar (may be time consuming)
    if progress_bar:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            loss_dataset = mapped.compute()
    else:
        loss_dataset = mapped.compute()
    return loss_dataset

def dask_loss(loss_fn):
    """
    A decorator (wrapper) for dask computations, used in conjunction to the function compute_loss_manifold().
    This decorator wraps a loss function with inputs: loss_fn(subspace, measurements, **kwargs).
    The subspace is computed by: subspace = eigenvectors * eigenvalues. It is deleted (and garbage collected)
    to avoid memory overload. The resulting residual dimensions are expanded to the manifold dimensions.


    Returns
    -------
    wrapper: pynoisy.inverse.dask_loss,
        A wrapped loss function which takes care of dask related tasks with some post (and pre) processing.
    """
    def wrapper(modes, measurements, dim_names, **kwargs):
        subspace = modes.eigenvalues * modes.eigenvectors
        residual = loss_fn(subspace, measurements, **kwargs)

        # Expand dimensions
        dims = dict([(dim, subspace[dim].data) for dim in dim_names])
        residual = residual.expand_dims(dims)

        # Release memory
        del subspace
        gc.collect()

        return residual
    return wrapper

@dask_loss
def pixel_loss_fn(subspace, measurements, damp=0.0):
    """
    Compute projection residual of the *direct pixel measurements* onto the subspace.
    The function solves ``min ||Ax - b||^2`` or the damped version: ``min ||Ax - b||^2 + d^2 ||x||^2``,
    where A is the subspace matrix, b is the input vector and d is the damping factor.

    Parameters
    ----------
    subspace: xr.DataArray,
        A DataArray with the spanning vectors along dimension 'degree'.
        Note that for low rank approximation of a matrix the subspace should be the multiplication:
        eigenvectors * eigenvalues.
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
    """
    subspace = subspace.where(np.isfinite(measurements))
    loss = pynoisy.linalg.projection_residual(measurements, subspace, damp=damp)
    return loss
