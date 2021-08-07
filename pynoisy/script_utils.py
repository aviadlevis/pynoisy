"""
Utility functions and methods used across scripts
"""
import numpy as _np
import pynoisy
import xarray as _xr
import time as _time
import functools as _functools
import itertools as _itertools
import os as _os

def estimate_envelope_mem(obs, movie_coords, total_flux=1.0, fwhm=80.0, FluxReg_w=1e5, MEMReg_w=2e5,
                          output_path=None):

    from scipy.optimize import minimize

    # Datafit Operators
    ehtOp = pynoisy.operators.ObserveOp(obs, movie_coords)
    modulateOp = pynoisy.operators.ModulateOp(_xr.DataArray(1.0, coords=movie_coords))
    forwardOp = ehtOp * modulateOp
    data_ops = pynoisy.operators.L2LossOp(obs.data['vis'], forwardOp, obs.data['sigma'])

    # Regularization Operators
    fov = movie_coords.to_dataset().utils_image.fov
    ny, nx = movie_coords['y'].size, movie_coords['x'].size
    prior_image = pynoisy.envelope.gaussian(ny, nx, fov=fov, fwhm=fwhm, total_flux=total_flux)
    reg_ops = [
        pynoisy.operators.FluxRegOp(prior=total_flux, weight=FluxReg_w),
        pynoisy.operators.MEMRegOp(prior=prior_image, weight=MEMReg_w)
    ]

    # Define loss function, initial guess and bounds
    lossOp = pynoisy.operators.Loss(data_ops=data_ops, reg_ops=reg_ops)
    x0 = _np.zeros(ny * nx)
    bounds = [(0, None) for _ in range(x0.size)]

    # Use scipy minimize to estimate the envelope
    output = minimize(lossOp, x0=x0, jac=lossOp.jac, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'disp': True})

    envelope = _xr.DataArray(output['x'].reshape(ny, nx), dims=['y', 'x'],
                            coords={'y': movie_coords['y'], 'x': movie_coords['x']})
    envelope.attrs.update(total_flux=total_flux, fwhm=fwhm, niter=output['nit'])
    for op in reg_ops:
        envelope.attrs.update({type(op).__name__: op.w})
    envelope.name = 'envelope_estimate'

    if output_path:
        mode = 'a' if _os.path.exists(output_path) else 'w'
        envelope.to_netcdf(output_path, mode=mode)
    return envelope

def observations_from_correlation_angles(temporal_angle, spatial_angle, eht_array, tint=60.0,
                                         tstart=4.0, tstop=15.5,ntobs=64, nt=64, ny=64, nx=64,
                                         envelope_model=pynoisy.envelope.ring, total_flux=1.0,
                                         fov=(160.0, 'uas'), alpha=2.0, thermal_noise=True,
                                         frac_noise=0.05, seed=None, output_path=None):

    import ehtim as eh

    # Generate a stochastic video by modulating an envelope with a Gaussian Random Field.
    # Randomly sample the *true* underlying accretion parameters.
    diffusion = pynoisy.diffusion.general_xy(ny=ny, nx=nx, opening_angle=spatial_angle)
    advection = pynoisy.advection.general_xy(ny=ny, nx=nx, opening_angle=temporal_angle)
    solver = pynoisy.forward.HGRFSolver(advection, diffusion, nt=nt, seed=seed)
    grf = solver.run().utils_image.set_fov(fov).utils_movie.set_time(tstart=tstart, tstop=tstop, units='UTC')

    # Generate source movie from envelope and GRF
    envelope = envelope_model(ny=ny, nx=nx, total_flux=total_flux).utils_image.set_fov(fov)
    source = pynoisy.forward.modulate(envelope, grf, alpha)

    # Load EHT array and generate an empty Observation object
    # Generate complex visibility meausrements with thermal noise added.
    array = eh.array.load_txt(eht_array)
    obs = pynoisy.observation.empty_eht_obs(array, nt=ntobs, tint=tint)
    obs = pynoisy.observation.observe_same(source, obs, thermal_noise=thermal_noise)
    if frac_noise > 0:
        obs = obs.add_fractional_noise(frac_noise)

    source_data = _xr.merge([grf, envelope, source])
    if output_path:
        solver.to_netcdf(output_path, group='solver')
        source_data.to_netcdf(output_path, mode='a')
    return obs, source_data


def netcdf_attrs(attrs_dict, add_datetime=True, add_github_version=True):
    """
    NetCDF attributes doesnt save boolean or None attributes.

    Parameters
    ----------
    attrs_dict: dict,
        Attribute dictionary
    add_datetime: bool, default=True
        Add the current date and time
    add_github_version: bool, default=True
        Add the current date and time

    Returns
    -------
    attrs: dict,
        Attributes that can be saved to NetCDF format

    Raises
    ------
    warning if there are uncomitted changes in pynoisy or inoisy.
    """
    attrs = dict()
    if add_datetime:
        attrs['date'] = _time.strftime("%d-%b-%Y-%H:%M:%S")
    if add_github_version:
        attrs['github_version'] = pynoisy.utils.github_version()

    for key in attrs_dict.keys():
        attrs[key] = str(attrs_dict[key]) if (isinstance(attrs_dict[key], bool) or (attrs_dict[key] is None)) \
                     else attrs_dict[key]
    return attrs

def get_parameter_grid(config):
    """
    Generate parameter grid according to `variable_params` in the configuration file.

    Parameters
    ----------
    config: dict,
        Dictionary loaded from yaml file.

    Returns
    -------
    param_grid: list,
        A list of parameters spanning the range of combinations of `variable_params`.
    """
    param_grid = []
    num_parameters = 0
    for field_type, params in config.items():
        if 'variable_params' in params:
            for param, grid_spec in config[field_type]['variable_params'].items():
                if ('range' in grid_spec.keys()) and ('num' in grid_spec.keys()):
                    # Linearly spaced grid defined by `range` and `num`
                    grid = _np.linspace(*grid_spec['range'], grid_spec['num'])
                elif 'values' in grid_spec:
                    grid = grid_spec['values']
                else:
                    raise AttributeError
                param_grid.append([{field_type: {param: value}} for value in grid])
                num_parameters += 1

    param_grid = [_functools.reduce(lambda x, y: {**x, **y}, element) for element in \
                  _itertools.product(*param_grid)]
    return param_grid

def expand_dataset_dims(dataset, config, parameters):
    """
    Expand datset to include configuration file dimensions

    Parameters
    ----------
    dataset: xr.Dataset,
        An xarray dataset
    config: dict,
        Dictionary loaded from yaml file.
    parameters: dict,
        Dictionary with 'variable_params' from configuration file.
    """
    for field_type, params in parameters.items():
        for param, value in params.items():
            dataset = dataset.expand_dims({config[field_type]['variable_params'][param]['dim_name']: [value]})
    return dataset

def get_default_solver(config, variable_params={}):
    """
    Generate an HGRFSolver from the `fixed_params` in the configuration file.

    Parameters
    ----------
    config: dict,
        Dictionary loaded from yaml file.
    variable_params: dict, optional
        A dictionary containing 'diffusion' and / or 'advection' keys with relevant parameters.

    Returns
    -------
    solver: pynoisy.forward.HGRFSolver,
        A solver generated from default, fixed and variable parameters.
    """
    nt, ny, nx = config['grid']['nt'], config['grid']['ny'], config['grid']['nx']
    diffusion_model = getattr(pynoisy.diffusion, config['diffusion']['model'])
    advection_model = getattr(pynoisy.advection, config['advection']['model'])
    solver_model =  getattr(pynoisy.forward, config['solver']['model'])
    advection = advection_model(ny, nx, **config['advection'].get('fixed_params', {}),
                                **variable_params.get('advection', {}))
    diffusion = diffusion_model(ny, nx, **config['diffusion'].get('fixed_params', {}),
                                **variable_params.get('diffusion', {}))
    solver = solver_model(advection, diffusion, nt, **config['solver'].get('fixed_params', {}),
                          **variable_params.get('solver', {}))
    return solver

def get_regularization_ops(reg_params):
    """
    Generate regularization Operators for the regularization param dictionary.

    Parameters
    ----------
    reg_params: dict,
        Dictionary loaded from yaml file.

    Returns
    -------
    reg_ops: list,
        list of regularization operators.
    """
    reg_ops = []
    for reg_op, reg_value in reg_params.items():
        if reg_op == 'MEMRegOp':
            model = reg_value['prior_image']['model']
            prior_model = getattr(pynoisy.envelope, model)
            prior_image = prior_model(**reg_value['prior_image']['fixed_params'])
            reg_ops.append(pynoisy.operators.MEMRegOp(prior=prior_image, weight=reg_value['weight']))
        else:
            reg_model = getattr(pynoisy.operators, reg_op)
            reg_ops.append(reg_model(**reg_value))
    return reg_ops

