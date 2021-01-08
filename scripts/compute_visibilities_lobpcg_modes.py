import xarray as xr
from pynoisy import eht_functions as ehtf
from tqdm import tqdm
import glob
import os
import pynoisy.utils
import argparse

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--array_year',
                        default='2017',
                        type=int,
                        help='(default value: %(default)s) Array file year.')
    parser.add_argument('--target',
                        default='sgra',
                        help='(default value: %(default)s) Target (used for ra and dec): e.g. sgra or m87.')
    parser.add_argument('--fov',
                         type=float,
                         default=140.0,
                         help='(default value: %(default)s) Measurement field of view.')
    parser.add_argument('--directory',
                        default='opening_angle_modes',
                        help='(default value: %(default)s) Path to input / output directory.')
    parser.add_argument('--startswith',
                        default='modes',
                        help='(default value: %(default)s) Modes file names start with this string.')
    args = parser.parse_args()
    return args

# Parse input arguments
args = parse_arguments()

if args.target == 'sgra':
    uvfits_path = '/home/aviad/Code/eht-imaging/SgrA/data/ER6_may2020/hops/hops_3601_SGRA_lo_V0_both_scan_netcal_LMTcal_normalized_10s.uvfits'
else:
    raise AttributeError('target: {} not implemented'.format_map(args.target))

# fov = 250.0
array_path = '/home/aviad/Code/eht-imaging/arrays/EHT{}.txt'.format(args.array_year)
obs = ehtf.load_obs(array_path, uvfits_path)

directory = './opening_angle_modes_12-05-2020/'
files = [file for file in glob.glob(os.path.join(args.directory, '*.nc')) if file.split('/')[-1].startswith(args.startswith)]

for file in tqdm(files, desc='file'):
    modes = xr.load_dataset(file)
    eigenvectors = modes.eigenvectors
    modes_angle = []
    for angle in tqdm(eigenvectors.temporal_angle, desc='angle', leave=False):
        modes_deg = []
        for deg in eigenvectors.deg:
            movie = ehtf.xarray_to_hdf5(eigenvectors.sel(deg=deg, temporal_angle=angle), obs, args.fov, flipy=True)
            obs_data = movie.observe_same_nonoise(obs)
            data = xr.Dataset(data_vars={'vis': ('index', obs_data.data['vis']),
                                         'sigma': ('index', obs_data.data['sigma'])})
            data = data.expand_dims({'deg': [deg], 'temporal_angle': [angle],
                                     'spatial_angle': eigenvectors.spatial_angle})
            modes_deg.append(data)
        modes_angle.append(xr.concat(modes_deg, dim='deg'))
    visibility_modes = xr.concat(modes_angle, dim='temporal_angle')
    visibility_modes.attrs = modes.attrs
    visibility_modes.attrs.update({
        'modes_nt': modes.t.size,
        'modes_nx': modes.x.size,
        'modes_ny': modes.y.size,
        'fov': args.fov,
        'array_path': array_path,
        'uvfits_path': uvfits_path
    })
    pynoisy.utils.save_complex(visibility_modes, path=file.replace('modes.', 'vismodes.fov{}.{}array.{}'.format(
        args.fov, args.array_year, args.target)))
