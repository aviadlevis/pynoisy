"""
TODO: Some documentation and general description goes here.
"""
import xarray as xr
import numpy as np
import noisy_core
import pynoisy.utils as utils
from joblib import Parallel, delayed
from tqdm import tqdm

class NoisySolver(object):
    """TODO"""
    def __init__(self, advection, diffusion, forcing_strength=1.0, seed=None):
        self._advection = advection.copy(deep=True)
        self._diffusion = diffusion.copy(deep=True)
        self._params = utils.get_grid()
        self._params.update({'forcing_strength': forcing_strength,
                             'seed': None, 'num_frames': noisy_core.get_num_frames()})
        self._params.attrs = {'solver_type': 'Noisy'}
        self.reseed(seed)

    def copy(self, deep=True):
        """TODO"""
        return NoisySolver(
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

    def run_asymmetric(self, evolution_length=0.1, verbose=True, num_samples=1, n_jobs=1, seed=None):
        """TODO"""

        if seed is not None:
            self.reseed(seed)

        sample_range = tqdm(range(num_samples)) if verbose is True else range(num_samples)
        if n_jobs == 1:
            pixels = [noisy_core.run_asymmetric(
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
                    self.diffusion.tensor_ratio,
                    self.forcing_strength, evolution_length,
                    np.array(self.diffusion.principle_angle, dtype=np.float64, order='C'),
                    np.array(self.v, dtype=np.float64, order='C'),
                    np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                    np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                    False, self.seed + i) for i in sample_range)

        movie = xr.DataArray(data=pixels,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, self.num_frames),
                                     'x': self.params.x, 'y': self.params.y},
                             dims=['sample', 't', 'x', 'y'])
        if num_samples == 1:
            movie.attrs['seed'] = self.seed
            movie = movie.squeeze('sample')
        else:
            movie.coords['seed'] = ('sample', self.seed + np.arange(num_samples))
        return movie

    def symmetric_source(self, evolution_length=0.1, num_samples=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)
        source = np.random.randn(num_samples, self.num_frames, *noisy_core.get_image_size()) * self.forcing_strength
        movie = xr.DataArray(data=source,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, self.num_frames),
                                     'x': self.params.x, 'y': self.params.y},
                             dims=['sample', 't', 'x', 'y'])
        if num_samples == 1:
            movie = movie.squeeze('sample')
        return movie

    def run_symmetric(self, source=None, evolution_length=0.1, verbose=True, num_samples=1, n_jobs=1, seed=None):
        """TODO"""

        if seed is not None:
            self.reseed(seed)

        # Either pre-determined source or randomly sampled from Gaussian distribution
        if source is None:
            source_attr = 'random (numpy)'
            np.random.seed(self.seed)
            source = np.random.randn(num_samples, self.num_frames, *noisy_core.get_image_size()) * self.forcing_strength
        else:
            source_attr = 'controlled (input)'
            source_type = type(source)
            if source_type == np.ndarray:
                source = np.expand_dims(source, 0) if source.ndim==3 else source
                num_samples = source.shape[0]
            elif source_type == xr.DataArray:
                source = source.expand_dims('sample') if 'sample' not in source.dims else source
                num_samples = source.sample.size
            else:
                raise AttributeError('Source type ({}) not implemented'.format(source_type))

        sample_range = tqdm(range(num_samples)) if verbose is True else range(num_samples)

        if n_jobs == 1:
            pixels = [noisy_core.run_symmetric(
                np.array(self.diffusion.tensor_ratio, dtype=np.float64, order='C'),
                evolution_length, np.array(self.diffusion.principle_angle, dtype=np.float64, order='C'),
                np.array(self.v, dtype=np.float64, order='C'),
                np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                np.array(source[i], dtype=np.float64, order='C'), verbose) for i in sample_range]
        else:
            pixels = Parallel(n_jobs=min(n_jobs, num_samples))(
                delayed(noisy_core.run_symmetric)(
                np.array(self.diffusion.tensor_ratio, dtype=np.float64, order='C'),
                evolution_length, np.array(self.diffusion.principle_angle, dtype=np.float64, order='C'),
                np.array(self.v, dtype=np.float64, order='C'),
                np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                np.array(source[i], dtype=np.float64, order='C'), verbose) for i in sample_range)

        movie = xr.DataArray(data=pixels,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, self.num_frames),
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
            self.diffusion.tensor_ratio,
            np.array(self.diffusion.principle_angle, dtype=np.float64, order='C'),
            np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
            np.array(self.v, dtype=np.float64, order='C'),
            np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
            np.array(movie, dtype=np.float64, order='C')
        )
        laplacian = xr.DataArray(data=lap, coords=movie.coords, dims=movie.dims)
        return laplacian

    def adjoint_angle_derivative(self, forward, adjoint):
        gradient = noisy_core.adjoint_angle_derivative(
            self.diffusion.tensor_ratio,
            np.array(self.diffusion.principle_angle, dtype=np.float64, order='C'),
            np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
            np.array(self.v, dtype=np.float64, order='C'),
            np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
            np.array(forward, dtype=np.float64, order='C'),
            np.array(adjoint, dtype=np.float64, order='C')
        )
        gradient = xr.DataArray(data=gradient, coords=self.params.coords, dims=self.params.dims)
        return gradient

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
    def seed(self):
        return int(self.params.seed)

    @property
    def num_frames(self):
        return int(self.params.num_frames)

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

