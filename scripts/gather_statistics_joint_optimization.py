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
    parser.add_argument('--n_jobs',
                         type=int,
                         default=4,
                         help='(default value: %(default)s) Number of jobs.')
    parser.add_argument('--seed',
                         type=int,
                         default=27669,
                         help='(default value: %(default)s) measurements seed.')
    parser.add_argument('--degree',
                        type=int,
                        default=np.inf,
                        help='(default value: %(default)s) Modes degree.')
    parser.add_argument('--num_grid',
                         type=int,
                         default=5,
                         help='(default value: %(default)s) num_grid**2 number of true opening angle points.')
    parser.add_argument('--damp',
                        default=1.0,
                        type=float,
                        help='(default value: %(default)s) Amplitude of regularization (dampning).')
    parser.add_argument('--fov',
                        default=160.0,
                        type=float,
                        help='(default value: %(default)s) Field of view (uas).')
    parser.add_argument('--alpha',
                        default=3.0,
                        type=float,
                        help='(default value: %(default)s) Variance of the GRF.')
    parser.add_argument('--fft_pad_factor',
                        default=1,
                        type=float,
                        help='(default value: %(default)s) Variance of the GRF.')
    parser.add_argument('--tint',
                        default=60.0,
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

def generate_sparse_measurements(temporal_angle, spatial_angle, obs, alpha, seed, fft_pad_factor, fov):
    advection = pynoisy.advection.general_xy(nx, ny, opening_angle=float(temporal_angle))
    diffusion = pynoisy.diffusion.general_xy(nx, ny, opening_angle=float(spatial_angle))
    solver = pynoisy.forward.HGRFSolver(nx, ny, advection, diffusion, seed=seed)
    grf = solver.run(num_frames=nt, n_jobs=4, verbose=False)
    envelope = pynoisy.envelope.ring(nx, ny, inner_radius=0.17, inner_decay=8)

    movie = (np.exp(alpha*grf) * envelope).noisy_methods.to_world_coords(tstart=obs.tstart, tstop=obs.tstop, fov=fov)
    movie.attrs.update(grf.attrs)
    movie.attrs.update(alpha=alpha)
    measurements = ehtf.compute_block_visibilities(movie, psize=movie.psize, fft_pad_factor=fft_pad_factor)
    measurements = pynoisy.utils.sample_eht(measurements, obs)
    envelope = envelope.noisy_methods.to_world_coords(tstart=obs.tstart, tstop=obs.tstop, fov=fov)
    grf = grf.noisy_methods.to_world_coords(tstart=obs.tstart, tstop=obs.tstop, fov=fov)
    return measurements, envelope, movie, grf

def load_obs(array_year, tint, nt):
    uvfits_path = '/home/aviad/Code/eht-imaging/SgrA/data/ER6_may2020/casa/casa_3599_SGRA_lo_netcal_LMTcal_normalized_10s.uvfits'
    array_path = '/home/aviad/Code/eht-imaging/arrays/'

    if array_year == 2017:
        array_path += 'EHT2017_m87.txt'
    elif array_year == 2025:
        array_path += 'EHT2025.txt'

    array = eh.array.load_txt(array_path)
    obs_sgra = eh.obsdata.load_uvfits(uvfits_path, remove_nan=True)

    tadv = (obs_sgra.tstop - obs_sgra.tstart) * 3600.0 / nt
    obs = ehtf.array_to_obs(array, obs_sgra, tadv, tint)
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


# Parse input arguments
args = parse_arguments()

seed = args.seed
num_grid = args.num_grid
fft_pad_factor = args.fft_pad_factor
fov = args.fov
alpha = args.alpha
tint = args.tint
startswith = args.startswith
damp=args.damp

files = [file for file in glob.glob(os.path.join(args.input_dir, '*.nc')) if file.split('/')[-1].startswith(startswith)]
modes = pynoisy.utils.load_modes(files[0])
nt, nx, ny =  modes.t.size, modes.x.size, modes.y.size

true_spatial_angle = np.linspace(-1.2, 1.2, num_grid)
true_temporal_angle = np.linspace(-3.14, 3.14, num_grid, endpoint=False)

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

for array_year in [2017]:
    obs = load_obs(array_year=array_year, tint=tint, nt=nt)

    residual_stats, envelope_stats = [], []
    for spatial_angle in tqdm(true_spatial_angle, desc='true spatial angle'):
        residuals, envelope_estimates = [], []
        for temporal_angle in tqdm(true_temporal_angle, desc='true temporal angle'):
            measurements, envelope, movie, grf_true = generate_sparse_measurements(temporal_angle, spatial_angle, obs,
                                                                                   alpha, seed, fft_pad_factor, fov)
            ehtop = pynoisy.inverse.EHTOperator(measurements, envelope.coords, fft_pad_factor)
            data = measurements.data[np.isfinite(measurements.data)]

            envelope_estimate_iter, res_iter = [], []
            for iteration in range(args.num_iter):
                xinv = pylops.optimization.sparsity.SplitBregman(ehtop, Dop, data,
                                                                 niter, niterinner,
                                                                 mu=mu, epsRL1s=epsRL1s,
                                                                 tol=tol, tau=tau,
                                                                 **dict(iter_lim=5, damp=1e-4))[0]

                envelope_estimate = xr.DataArray(np.real(xinv.reshape(nx, ny)), coords=envelope.coords).clip(0.0)
                envelope_estimate_iter.append(envelope_estimate.expand_dims(iteration=[iteration]))

                res, coef = pynoisy.utils.opening_angles_vis_residuals(
                    files, measurements, obs, envelope_estimate, damp=damp, fft_pad_factor=fft_pad_factor, return_coefs=True)
                res_iter.append(res.data.expand_dims(iteration=[iteration]))

                projection = estimate_grf_lowrank(measurements, res.data, coef)
                ehtop.set_grf(projection)

            envelope_estimate_iter = xr.concat(envelope_estimate_iter, dim='iteration')
            res_iter = xr.concat(res_iter, dim='iteration')

            residuals.append(res_iter.expand_dims({'true_temporal_angle': [temporal_angle],
                                                   'true_spatial_angle': [spatial_angle]}))
            envelope_estimates.append(envelope_estimate_iter.expand_dims({'true_temporal_angle': [temporal_angle],
                                                                          'true_spatial_angle': [spatial_angle]}))
        residual_stats.append(xr.concat(residuals, dim='true_temporal_angle'))
        envelope_stats.append(xr.concat(envelope_estimates, dim='true_temporal_angle'))
    residual_stats = xr.concat(residual_stats, dim='true_spatial_angle')
    envelope_stats = xr.concat(envelope_stats, dim='true_spatial_angle')

    # update attributes
    envelope_stats.attrs.update(
        desc='Envelope estimate using total variation regularization',
        modes_dir=args.input_dir,
        tvx_eps=epsRL1s[0],
        tvy_eps=epsRL1s[1],
        niter=niter,
        niterinner=niterinner,
        tau=tau,
        tol=tol,
        envelope_inner_radius=envelope.inner_radius,
        envelope_inner_decay=envelope.inner_decay,
        fft_pad_factor=fft_pad_factor,
        fov=fov,
        alpha=alpha,
        tint=tint)

    residual_stats.attrs.update(
            desc='Sparse sampled residuals estimated envelope and linearization',
            modes_dir=args.input_dir,
            damp=damp,
            array_year=array_year,
            envelope_inner_radius=envelope.inner_radius,
            envelope_inner_decay=envelope.inner_decay,
            fft_pad_factor=fft_pad_factor,
            fov=fov,
            alpha=alpha,
            tint=tint)

    # Save output NetCDF
    residual_stats.to_netcdf(
        os.path.join(args.output_dir, 'jointopt.residuals.eht{}.damp{:1.2f}.stats.num_spatial{}.num_temporal{}.seed{}.degree{}.nc'.format(
            array_year, damp, residual_stats.true_spatial_angle.size, residual_stats.true_temporal_angle.size, seed, int(residual_stats.deg))))

    envelope_stats.to_netcdf(
        os.path.join(args.output_dir, 'jointopt.envelope.eht{}.stats.num_spatial{}.num_temporal{}.seed{}.degree{}.nc'.format(
             array_year, residual_stats.true_spatial_angle.size, residual_stats.true_temporal_angle.size, seed, int(residual_stats.deg))))
