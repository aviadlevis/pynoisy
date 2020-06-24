"""
TODO: Some documentation and general description goes here.
"""
import xarray as xr
import numpy as np
import noisy_core, hgrf_core
import pynoisy.utils as utils
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings


class NoisySolver(object):
    """TODO"""
    def __init__(self, nx, ny, advection, diffusion, forcing_strength=1.0, seed=None):
        warnings.simplefilter(action='ignore', category=FutureWarning)
        grid = utils.get_grid(nx, ny)
        resampled_advection = advection.interp_like(grid).bfill(dim='x').ffill(dim='x').bfill(dim='y').ffill(dim='y')
        resampled_advection.coords.update(grid)
        resampled_diffusion = diffusion.interp_like(grid).bfill(dim='x').ffill(dim='x').bfill(dim='y').ffill(dim='y')
        resampled_diffusion.coords.update(grid)
        self._advection = resampled_advection
        self._diffusion = resampled_diffusion
        self._params = grid
        self._params.update({'forcing_strength': forcing_strength, 'seed': None})
        self._params.attrs = {'solver_type': 'Noisy'}
        self.reseed(seed)
        warnings.resetwarnings()

    def copy(self, deep=True):
        """TODO"""
        return NoisySolver(self.nx, self.ny,
            advection=self.advection.copy(deep=deep), diffusion=self.diffusion.copy(deep=deep),
            forcing_strength=self.forcing_strength, seed=self.seed)

    def reseed(self, seed=None):
        seed = np.random.randint(0, 32767) if seed is None else seed
        self._params['seed'] = seed
        print('Setting solver seed to: {}'.format(self.seed), end='\r')

    def update_advection(self, advection):
        """TODO"""
        self._advection.update(advection)
        self._advection.attrs = advection.attrs

    def update_diffusion(self, diffusion):
        """TODO"""
        self._diffusion.update(diffusion)
        self._diffusion.attrs = diffusion.attrs

    def run_asymmetric(self, num_frames, evolution_length=0.1, verbose=True, num_samples=1, n_jobs=1, seed=None):
        """TODO"""

        if seed is not None:
            self.reseed(seed)

        sample_range = tqdm(range(num_samples)) if verbose is True else range(num_samples)
        if n_jobs == 1:
            pixels = [noisy_core.run_asymmetric(
                num_frames, self.nx, self.ny,
                self.diffusion.tensor_ratio,
                self.forcing_strength, evolution_length,
                np.array(self.diffusion.principle_angle, dtype=np.float64, order='C'),
                np.array(self.v, dtype=np.float64, order='C'),
                np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                verbose, self.seed) for i in sample_range]

        else:
            pixels = Parallel(n_jobs=min(n_jobs, num_samples))(
                delayed(noisy_core.run_asymmetric)(
                    num_frames, self.nx, self.ny,
                    self.diffusion.tensor_ratio,
                    self.forcing_strength, evolution_length,
                    np.array(self.diffusion.principle_angle, dtype=np.float64, order='C'),
                    np.array(self.v, dtype=np.float64, order='C'),
                    np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                    np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                    False, self.seed + i) for i in sample_range)

        movie = xr.DataArray(data=pixels,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, num_frames),
                                     'x': self.params.x, 'y': self.params.y},
                             dims=['sample', 't', 'x', 'y'])
        if num_samples == 1:
            movie.attrs['seed'] = self.seed
            movie = movie.squeeze('sample')
        else:
            movie.coords['seed'] = ('sample', self.seed + np.arange(num_samples))
        return movie

    def symmetric_source(self, num_frames, evolution_length=0.1, num_samples=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)
        source = np.random.randn(num_samples, num_frames, self.nx, self.ny) * self.forcing_strength
        movie = xr.DataArray(data=source,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, num_frames),
                                     'x': self.params.x, 'y': self.params.y},
                             dims=['sample', 't', 'x', 'y'])
        if num_samples == 1:
            movie = movie.squeeze('sample')
        return movie

    def run_symmetric(self, source=None, evolution_length=0.1, verbose=True, num_frames=None, num_samples=1, n_jobs=1, seed=None):
        """TODO"""

        if seed is not None:
            self.reseed(seed)

        # Either pre-determined source or randomly sampled from Gaussian distribution
        if source is None:
            assert num_frames is not None, 'If the source is unspecified, the number of frames should be specified.'
            source_attr = 'random (numpy)'
            np.random.seed(self.seed)
            source = np.random.randn(num_samples, num_frames, self.nx, self.ny) * self.forcing_strength
        else:
            source_attr = 'controlled (input)'
            source_type = type(source)
            if source_type == np.ndarray:
                source = np.expand_dims(source, 0) if source.ndim==3 else source
                num_samples = source.shape[0]
                num_frames = source.shape[1]
            elif source_type == xr.DataArray:
                source = source.expand_dims('sample') if 'sample' not in source.dims else source
                num_samples = source.sample.size
                num_frames = source.t.size
            else:
                raise AttributeError('Source type ({}) not implemented'.format(source_type))

        sample_range = tqdm(range(num_samples)) if verbose is True else range(num_samples)

        if n_jobs == 1:
            pixels = [noisy_core.run_symmetric(
                num_frames, self.nx, self.ny,
                np.array(self.diffusion.tensor_ratio, dtype=np.float64, order='C'),
                evolution_length, np.array(self.diffusion.principle_angle, dtype=np.float64, order='C'),
                np.array(self.v, dtype=np.float64, order='C'),
                np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                np.array(source[i], dtype=np.float64, order='C'), verbose) for i in sample_range]
        else:
            pixels = Parallel(n_jobs=min(n_jobs, num_samples))(
                delayed(noisy_core.run_symmetric)(
                    num_frames, self.nx, self.ny,
                    np.array(self.diffusion.tensor_ratio, dtype=np.float64, order='C'),
                    evolution_length, np.array(self.diffusion.principle_angle, dtype=np.float64, order='C'),
                    np.array(self.v, dtype=np.float64, order='C'),
                    np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                    np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                    np.array(source[i], dtype=np.float64, order='C'), verbose) for i in sample_range)

        movie = xr.DataArray(data=pixels,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, num_frames),
                                     'x': self.params.x, 'y': self.params.y},
                             dims=['sample', 't', 'x', 'y'])

        movie.attrs['source'] = source_attr
        movie.attrs['seed'] = self.seed if source_attr=='random (numpy)' else None
        if num_samples == 1:
            movie = movie.squeeze('sample')

        return movie

    def run_adjoint(self, source, evolution_length=0.1, verbose=True, num_samples=1, n_jobs=1):
        advection = self.advection
        self.update_advection(-advection)
        adjoint = self.run_symmetric(source, evolution_length, verbose, num_samples, n_jobs)
        self.update_advection(advection)
        return adjoint

    def get_laplacian(self, movie):
        lap = noisy_core.get_laplacian(
            movie.shape[0], self.nx, self.ny,
            self.diffusion.tensor_ratio,
            np.array(self.diffusion.principle_angle, dtype=np.float64, order='C'),
            np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
            np.array(self.v, dtype=np.float64, order='C'),
            np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
            np.array(movie, dtype=np.float64, order='C')
        )
        laplacian = xr.DataArray(data=lap, coords=movie.coords, dims=movie.dims)
        return laplacian

    def save(self, path):
        self.params.to_netcdf(path, mode='w')
        self.advection.to_netcdf(path, group='advection', mode='a')
        self.diffusion.to_netcdf(path, group='diffusion', mode='a')

    @classmethod
    def from_netcdf(cls, path):
        params = xr.load_dataset(path)
        solver = cls(advection=xr.load_dataset(path, group='advection'),
                   diffusion=xr.load_dataset(path, group='diffusion'),
                   forcing_strength=params.forcing_strength.data, seed=params.seed.data)
        solver._params = params
        return solver

    @property
    def params(self):
        return self._params

    @property
    def nx(self):
        return self.params.x.size

    @property
    def ny(self):
        return self.params.y.size

    @property
    def seed(self):
        return int(self.params.seed)

    @property
    def forcing_strength(self):
        return float(self.params.forcing_strength)

    @property
    def diffusion(self):
        return self._diffusion

    @property
    def advection(self):
        return self._advection

    @property
    def v(self):
        return np.stack([self.advection.vx, self.advection.vy], axis=-1)

class HGRFSolver(object):
    def __init__(self, nx, ny, diffusion, solver_type='PCG', forcing_strength=1.0, seed=None):
        self._solver_list = ['PCG', 'SMG']
        assert solver_type in self._solver_list, 'Not supported solver type: {}'.format(solver_type)

        grid = utils.get_grid(nx, ny)
        resampled_diffusion = diffusion.interp_like(grid).bfill(dim='x').ffill(dim='x').bfill(dim='y').ffill(dim='y')
        resampled_diffusion.coords.update(grid)
        self._diffusion = resampled_diffusion
        self._solver_id = self._solver_list.index(solver_type)
        self._params = grid
        self._params.update({'solver_type': solver_type, 'forcing_strength': forcing_strength, 'seed': None})
        self._params.attrs = {'solver_type': 'HGRF'}
        self.reseed(seed)
        hgrf_core.init_mpi()

    def reseed(self, seed=None):
        seed = np.random.randint(0, 32767) if seed is None else seed
        self._params['seed'] = seed
        print('Setting solver seed to: {}'.format(self.seed), end='\r')

    def run(self, maxiter, evolution_length=0.1,
            source=None, verbose=2, num_frames=None, num_samples=1, n_jobs=1, seed=None):
        """TODO"""
        verbose = int(verbose)
        if seed is not None:
            self.reseed(seed)

        # Either pre-determined source or randomly sampled from Gaussian distribution
        if source is None:
            assert num_frames is not None, 'If the source is unspecified, the number of frames should be specified.'
            source_attr = 'random (numpy)'
            np.random.seed(self.seed)
            source = np.random.randn(num_samples, num_frames, self.nx, self.ny) * self.forcing_strength
        else:
            source_attr = 'controlled (input)'
            source_type = type(source)
            if source_type == np.ndarray:
                source = np.expand_dims(source, 0) if source.ndim == 3 else source
                num_samples = source.shape[0]
                num_frames = source.shape[1]
            elif source_type == xr.DataArray:
                source = source.expand_dims('sample') if 'sample' not in source.dims else source
                num_samples = source.sample.size
                num_frames = source.t.size
            else:
                raise AttributeError('Source type ({}) not implemented'.format(source_type))

        sample_range = tqdm(range(num_samples)) if verbose is True else range(num_samples)

        if n_jobs == 1:
            pixels = [hgrf_core.run(
                num_frames, self.nx, self.ny, self._solver_id, maxiter, verbose, self.diffusion.scaling_radius.data,
                self.diffusion.spatial_ratio.data, self.diffusion.temporal_ratio.data,
                np.array(source[i], dtype=np.float64, order='C'),
                np.array(self.diffusion.spatial_angle, dtype=np.float64, order='C'),
                np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                np.array(self.diffusion.correlation_length, dtype=np.float64, order='C')) for i in sample_range]
        else:
            pixels = Parallel(n_jobs=min(n_jobs, num_samples))(
                delayed(hgrf_core.run)(
                    num_frames, self.nx, self.ny, self._solver_id, maxiter, verbose, self.diffusion.scaling_radius.data,
                    self.diffusion.spatial_ratio.data, self.diffusion.temporal_ratio.data,
                    np.array(source[i], dtype=np.float64, order='C'),
                    np.array(self.diffusion.spatial_angle, dtype=np.float64, order='C'),
                    np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                    np.array(self.diffusion.correlation_length, dtype=np.float64, order='C')) for i in sample_range)
        movie = xr.DataArray(data=pixels,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, num_frames),
                                     'x': self.params.x, 'y': self.params.y},
                             dims=['sample', 't', 'x', 'y'])

        movie.attrs['source'] = source_attr
        movie.attrs['seed'] = self.seed if source_attr == 'random (numpy)' else None
        if num_samples == 1:
            movie = movie.squeeze('sample')

        return movie

    @property
    def params(self):
        return self._params

    @property
    def nx(self):
        return self.params.x.size

    @property
    def ny(self):
        return self.params.y.size

    @property
    def forcing_strength(self):
        return float(self.params.forcing_strength)

    @property
    def diffusion(self):
        return self._diffusion

    @property
    def seed(self):
        return int(self.params.seed)