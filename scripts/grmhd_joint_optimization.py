import xarray as xr
import pynoisy
import numpy as np
import argparse
from tqdm import tqdm
import os, glob
import ehtim as eh
import pylops
from pynoisy import eht_functions as ehtf

def parse_arguments():
    """Parse the command-line arguments for each run.

    Returns:
        args (Namespace): an argparse Namspace object with all the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',
                        default='extended_modes',
                        help='(default value: %(default)s) Path to input directory.')
    parser.add_argument('--output_dir',
                        default='extended_modes',
                        help='(default value: %(default)s) Path to output directory.')
    parser.add_argument('--startswith',
                        default='modes',
                        help='(default value: %(default)s) Modes file names start with this string.')
    parser.add_argument('--degree',
                        type=int,
                        default=np.inf,
                        help='(default value: %(default)s) Modes degree.')
    parser.add_argument('--damp',
                        default=0.0,
                        type=float,
                        help='(default value: %(default)s) Amplitude of regularization (dampning).')
    parser.add_argument('--fov',
                        default=160.0,
                        type=float,
                        help='(default value: %(default)s) Field of view (uas).')
    parser.add_argument('--fft_pad_factor',
                        default=1,
                        type=float,
                        help='(default value: %(default)s) Variance of the GRF.')
    parser.add_argument('--tint',
                        default=20.0,
                        type=float,
                        help='(default value: %(default)s) Integration time.')
    parser.add_argument('--tv',
                        default=1e3,
                        type=float,
                        help='(default value: %(default)s) TV regularization.')
    parser.add_argument('--num_iter',
                        default=3,
                        type=int,
                        help='(default value: %(default)s) Number of iterations alternating between envelope and dynamic parameter estimation.')
    args = parser.parse_args()
    return args

def generate_sparse_measurements(movie, obs, fft_pad_factor):
    measurements = ehtf.compute_block_visibilities(movie, psize=movie.psize, fft_pad_factor=fft_pad_factor)
    measurements = pynoisy.utils.sample_eht(measurements, obs)
    return measurements

def load_obs(obs_sgra, array_year, tstart, tstop, tint, tadv):
    array_path = '/home/aviad/Code/eht-imaging/arrays/'
    if array_year == 2017:
        array_path += 'EHT2017_m87.txt'
    elif array_year == 2025:
        array_path += 'EHT2025.txt'
    array = eh.array.load_txt(array_path)

    obs = ehtf.array_to_obs(array, obs_sgra, tadv, tint, tstart=tstart, tstop=tstop)
    return obs

def estimate_grf_lowrank(measurements, residuals, coefs):
    minimum = residuals[residuals.argmin(dim=['temporal_angle', 'spatial_angle'])].coords
    temporal_angle = float(minimum['temporal_angle'])
    spatial_angle = float(minimum['spatial_angle'])
    modes = pynoisy.utils.find_closest_modes(temporal_angle, spatial_angle, files)

    coefs_min = coefs.real.sel(temporal_angle=temporal_angle, spatial_angle=spatial_angle)
    projection = (coefs_min * modes.eigenvalues * modes.eigenvectors).sum('deg')
    projection = projection.noisy_methods.to_world_coords(
        tstart=float(measurements.t[0]), tstop=float(measurements.t[-1]), fov=measurements.fov)
    return projection


grmhd_paths = [
    '/home/aviad/Code/noisy/GRMHD/Ma+0.5_inc10.h5',
    '/home/aviad/Code/noisy/GRMHD/Ma0_inc10.h5',
    '/home/aviad/Code/noisy/GRMHD/Ma+0.94_inc10.h5'
]
# Parse input arguments
args = parse_arguments()

fft_pad_factor = args.fft_pad_factor
fov = args.fov
tint = args.tint
startswith = args.startswith
damp=args.damp

files = [file for file in glob.glob(os.path.join(args.input_dir, '*.nc')) if file.split('/')[-1].startswith(startswith)]
modes = pynoisy.utils.load_modes(files[0])
nt, nx, ny =  modes.t.size, modes.x.size, modes.y.size

# Total variation params
mu = 1.5
epsRL1s = [args.tv, args.tv]
niter = 20
niterinner = 5
tau=1.0
tol=1e-4

Dop = [
    pylops.FirstDerivative(ny * nx, dims=(nx, ny), dir=0, edge=False,
                           kind='backward', dtype=np.complex),
    pylops.FirstDerivative(ny * nx, dims=(nx, ny), dir=1, edge=False,
                           kind='backward', dtype=np.complex)
]

# Observation params
initial_frames = range(0,1000,64)[:-3]
uvfits_path = '/home/aviad/Code/eht-imaging/SgrA/data/ER6_may2020/casa/casa_3599_SGRA_lo_netcal_LMTcal_normalized_10s.uvfits'
obs_sgra = eh.obsdata.load_uvfits(uvfits_path, remove_nan=True)
tblock = (obs_sgra.tstop - obs_sgra.tstart) * (float(nt)/(initial_frames[-1] + nt))
tadv = tblock * 3600.0 / nt


for array_year in [2025]:
    for grmhd_path in grmhd_paths:
        grmhd_movie = pynoisy.utils.load_grmhd(grmhd_path)
        grmhd_movie = grmhd_movie.interp_like(pynoisy.utils.get_grid(nx, ny))

        tstart = obs_sgra.tstart
        tstop = tstart + tblock
        residuals, envelope_estimates = [], []
        for initial_frame in tqdm(initial_frames, desc='sliding window'):
            obs = load_obs(obs_sgra, array_year, tstart, tstop, tint, tadv)
            movie = grmhd_movie[initial_frame:initial_frame + nt]
            movie = movie.noisy_methods.to_world_coords(tstart=obs.tstart, tstop=obs.tstop, fov=fov)
            measurements = generate_sparse_measurements(movie, obs, fft_pad_factor)

            envelope_coords = xr.DataArray(dims=['x', 'y'], coords={'x': movie.x, 'y': movie.y}).coords
            ehtop = pynoisy.inverse.EHTOperator(measurements, envelope_coords, fft_pad_factor)

            data = measurements.data[np.isfinite(measurements.data)]

            envelope_estimate_iter, res_iter = [], []
            for iteration in range(args.num_iter):
                xinv = pylops.optimization.sparsity.SplitBregman(ehtop, Dop, data,
                                                                 niter, niterinner,
                                                                 mu=mu, epsRL1s=epsRL1s,
                                                                 tol=tol, tau=tau,
                                                                 **dict(iter_lim=5, damp=1e-4))[0]

                envelope_estimate = xr.DataArray(np.real(xinv.reshape(nx, ny)), coords=envelope_coords).clip(0.0)
                envelope_estimate_iter.append(envelope_estimate.expand_dims(iteration=[iteration]))

                res, coef = pynoisy.utils.opening_angles_vis_residuals(
                    files, measurements, envelope=envelope_estimate, damp=0.0, return_coefs=True
                )
                res_iter.append(res.data.expand_dims(iteration=[iteration]))

                projection = estimate_grf_lowrank(measurements, res.data, coef)
                ehtop.set_grf(projection)

            envelope_estimate_iter = xr.concat(envelope_estimate_iter, dim='iteration')
            res_iter = xr.concat(res_iter, dim='iteration')

            res_iter = res_iter.expand_dims(initial_frame=[initial_frame])
            res_iter = res_iter.assign_coords({'tstart': ('initial_frame', [tstart])})
            res_iter = res_iter.assign_coords({'tstop': ('initial_frame', [tstop])})

            envelope_estimate_iter = envelope_estimate_iter.expand_dims(initial_frame=[initial_frame])
            envelope_estimate_iter = envelope_estimate_iter.assign_coords({'tstart': ('initial_frame', [tstart])})
            envelope_estimate_iter = envelope_estimate_iter.assign_coords({'tstop': ('initial_frame', [tstop])})

            residuals.append(res_iter)
            envelope_estimates.append(envelope_estimate_iter)

            tstart += tblock
            tstop += tblock

        residuals = xr.concat(residuals, dim='initial_frame')
        envelope_estimates = xr.concat(envelope_estimates, dim='initial_frame')

        # update attributes
        envelope_estimates.attrs.update(
            desc='Envelope estimate using total variation regularization',
            grmhd=measurements.GRMHD,
            modes_dir=args.input_dir,
            tint=tint,
            array_year=array_year,
            fov=fov,
            tvx_eps=epsRL1s[0],
            tvy_eps=epsRL1s[1],
            niter=niter,
            niterinner=niterinner,
            tau=tau,
            tol=tol)

        residuals.attrs.update(
                desc='Sparse sampled residuals estimated envelope and linearization',
                grmhd=measurements.GRMHD,
                modes_dir=args.input_dir,
                damp=damp,
                array_year=array_year,
                fft_pad_factor=fft_pad_factor,
                fov=fov,
                tint=tint)

        # Save output NetCDF
        residuals.to_netcdf(
            os.path.join(args.output_dir, 'grmhd.{}.residuals.eht{}.damp{:1.2f}.degree{}.nc'.format(
                measurements.GRMHD, array_year, damp, int(residuals.deg))))

        envelope_estimates.to_netcdf(
            os.path.join(args.output_dir, 'grmhd.{}.envelope.eht{}.degree{}.nc'.format(
                measurements.GRMHD, array_year, int(residuals.deg))))
