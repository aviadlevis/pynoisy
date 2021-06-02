"""
Utility functions and methods used across scripts
"""
import numpy as _np
import pynoisy
import time as _time
import functools as _functools
import itertools as _itertools

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

