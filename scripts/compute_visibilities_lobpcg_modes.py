import xarray as xr
from pynoisy import eht_functions as ehtf
from tqdm import tqdm
import glob
import os
import pynoisy.utils

fov = 140.0

uvfits_path = '/home/aviad/Code/eht-imaging/SgrA/data/ER6_may2020/hops/hops_3601_SGRA_lo_V0_both_scan_netcal_LMTcal_normalized_10s.uvfits'
array_path = '/home/aviad/Code/eht-imaging/arrays/EHT2019.txt'
obs = ehtf.load_obs(array_path, uvfits_path)

directory = './opening_angles_modes/'

files = [file for file in glob.glob(os.path.join(directory, '*.nc')) \
         if file.split('/')[-1].startswith('modes')]

for file in tqdm(files, desc='file'):
    eigenvectors = xr.load_dataarray(file)
    modes_angle = []
    for angle in tqdm(eigenvectors.temporal_angle, desc='angle', leave=False):
        modes_deg = []
        for deg in eigenvectors.deg:
            movie = ehtf.xarray_to_hdf5(eigenvectors.sel(deg=deg, temporal_angle=angle), obs, fov, flipy=True)
            obs_data = movie.observe_same_nonoise(obs)
            data = xr.Dataset(data_vars={'vis': ('index', obs_data.data['vis']),
                                         'sigma': ('index', obs_data.data['sigma'])} )
            data = data.expand_dims({'deg': [deg], 'temporal_angle': [angle],
                                     'spatial_angle': eigenvectors.spatial_angle})
            modes_deg.append(data)
        modes_angle.append(xr.concat(modes_deg, dim='deg'))
    visibility_modes = xr.concat(modes_angle, dim='temporal_angle')
    visibility_modes.attrs = eigenvectors.attrs
    visibility_modes.attrs.update({
        'fov': fov,
        'array_path': array_path,
        'uvfits_path': uvfits_path
    })
    pynoisy.utils.save_complex(visibility_modes, path=file.replace('modes.', 'vismodes.flipped.2019array.'))
