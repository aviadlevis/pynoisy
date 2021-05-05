"""
This scripts computes and saves a dataset of Gaussian Random Fields (GRFs) [1,2].
This scripts takes a yaml configuration file as input which describes the
dataset parameters and gridding. See `configs/opening_angles.yaml` for an example configuration file.
To see description of command line arguments and help use
    `python compute_grfs_dataset.py --h

References
----------
.. [1] Lee, D. and Gammie, C.F., 2021. Disks as Inhomogeneous, Anisotropic Gaussian Random Fields.
   The Astrophysical Journal, 906(1), p.39. url: https://iopscience.iop.org/article/10.3847/1538-4357/abc8f3/meta
.. [2] inoisy code: https://github.com/AFD-Illinois/inoisy
"""
import pynoisy
import pynoisy.script_utils as script_utils
from tqdm import tqdm
import argparse, time, os
import shutil, ruamel_yaml
from pathlib import Path

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        help='Path to config yaml file.')
    parser.add_argument('--num',
                        default=10,
                        type=int,
                        help='(default value: %(default)s) Number of grfs.')

    # Parallel processing parameters
    parser.add_argument('--n_jobs',
                         type=int,
                         default=4,
                         help='(default value: %(default)s) Number of jobs for MPI processing of HYPRE.')
    parser.add_argument('--num_solvers',
                         type=int,
                         default=5,
                         help='(default value: %(default)s) Number of solvers used for RSI parallel processing.')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    with open(args.config, 'r') as stream:
        config = ruamel_yaml.load(stream, Loader=ruamel_yaml.Loader)

    # Datetime and github version information
    datetime = time.strftime("%d-%b-%Y-%H:%M:%S")
    github_version = pynoisy.utils.github_version()

    # Setup output path and directory
    dirpath = os.path.join(config['dataset']['outpath'].replace('<datetime>', datetime), 'grfs')
    Path(dirpath).mkdir(parents=True, exist_ok=True)

    # Setup NetCDF attributes
    attrs = script_utils.netcdf_attrs(vars(args))

    # Generate parameter grid according to configuration file
    param_grid = script_utils.get_parameter_grid(config)

    # Generate solver object with the fixed parameters
    solver = script_utils.get_default_solver(config, variable_params={'solver': {'num_solvers': args.num_solvers}})

    # Main iteration loop
    for i, params in enumerate(tqdm(param_grid, desc='parameter')):

        # Update solver object according to the variable parameters
        solver = script_utils.get_default_solver(config, params)

        grfs = solver.run(num_samples=args.num, n_jobs=args.n_jobs, verbose=0)
        grfs.name = 'grfs'

        # Expand modes dimensions to include the dataset parameters
        grfs = script_utils.expand_dataset_dims(grfs, config, params)

        # Save to datasets
        grfs.attrs.update(attrs)
        grfs = grfs.assign_coords({'file_index': i})
        grfs.to_netcdf(os.path.join(dirpath, 'grf{:04d}.nc'.format(i)))

    # Copy script and config for reproducibility
    shutil.copy(__file__, os.path.join(dirpath, 'script.py'.format(datetime)))
    shutil.copy(args.config, os.path.join(dirpath, 'config.yaml'.format(datetime)))