"""
TODO: Some documentation and general description goes here.
"""
import xarray as xr
import numpy as np
import core
from joblib import Parallel, delayed
from tqdm import tqdm

class NoisySolver(object):
    """TODO"""
    def __init__(self, advection=None, diffusion=None, coefficients=None, forcing_strength=1.0, seed=None):
        if coefficients is not None:
            self._coefficients = coefficients
        else:
            assert (advection is not None) and (diffusion is not None), 'Either coeffiecient (merge of advection and diffusion) or both advection and diffusion should be input'
            self._coefficients = xr.merge([advection, diffusion])
            self._coefficients.attrs['advection'] = advection.attrs
            self._coefficients.attrs['diffusion'] = diffusion.attrs
            self._coefficients.attrs['forcing_strength'] = forcing_strength
            self._coefficients.attrs['num_frames'] = core.get_num_frames()

        self._diffusion_vars = ['principle_angle', 'correlation_time', 'correlation_length', 'diffusion_coefficient']
        self._advection_vars = ['vx', 'vy']
        self.reseed(seed)

    def copy(self):
        return NoisySolver(coefficients=self.coefficients.copy())

    def reseed(self, seed=None):
        seed = np.random.randint(0, 32767) if seed is None else seed
        self.coefficients.attrs['seed'] = seed
        print('Setting solver seed to: {}'.format(self.seed), end='\r')

    def set_advection(self, advection):
        """TODO"""
        self._coefficients.update(advection)
        self._coefficients.attrs['advection'] = advection.attrs

    def set_diffusion(self, diffusion):
        """TODO"""
        self._coefficients.update(diffusion)
        self._coefficients.attrs['diffusion'] = diffusion.attrs

    def run_asymmetric(self, evolution_length=0.1, verbose=True, num_samples=1, n_jobs=1):
        """TODO"""
        sample_range = tqdm(range(num_samples)) if verbose is True else range(num_samples)
        if n_jobs == 1:
            pixels = [core.run_asymmetric(
                self.coefficients.attrs['diffusion']['tensor_ratio'],
                self.forcing_strength, evolution_length,
                np.array(self.coefficients.principle_angle, dtype=np.float64, order='C'),
                np.array(self.v, dtype=np.float64, order='C'),
                np.array(self.coefficients.diffusion_coefficient, dtype=np.float64, order='C'),
                np.array(self.coefficients.correlation_time, dtype=np.float64, order='C'),
                verbose, self.seed) for i in sample_range]

        else:
            pixels = Parallel(n_jobs=min(n_jobs, num_samples))(
                delayed(core.run_asymmetric)(
                    self.coefficients.attrs['diffusion']['tensor_ratio'],
                    self.forcing_strength, evolution_length,
                    np.array(self.coefficients.principle_angle, dtype=np.float64, order='C'),
                    np.array(self.v, dtype=np.float64, order='C'),
                    np.array(self.coefficients.diffusion_coefficient, dtype=np.float64, order='C'),
                    np.array(self.coefficients.correlation_time, dtype=np.float64, order='C'),
                    False, self.seed + i) for i in sample_range)

        movie = xr.DataArray(data=pixels,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, self.num_frames),
                                     'x': self.coefficients.x, 'y': self.coefficients.y},
                             dims=['sample', 't', 'x', 'y'])
        if num_samples == 1:
            movie.attrs['seed'] = self.seed
            movie = movie.squeeze('sample')
        else:
            movie.coords['seed'] = ('sample', self.seed + np.arange(num_samples))
        return movie

    def run_symmetric(self, source=None, evolution_length=0.1, verbose=True, num_samples=1, n_jobs=1):
        """TODO"""

        # Either pre-determined source or randomly sampled from Gaussian distribution
        if source is None:
            source_attr = 'random (numpy)'
            np.random.seed(self.seed)
            source = np.random.randn(num_samples, self.num_frames, *core.get_image_size()) * self.forcing_strength
        else:
            source_attr = 'controlled (input)'
            source_type = type(source)
            if source_type == np.ndarray:
                source = np.expand_dims(source, 0) if source.ndim==3 else source
                num_samples = source.shape[0]
            elif source_type == xr.DataArray:
                num_samples = source.sample.size
                source = source.expand_dims('sample') if 'sample' not in source.dims else source
            else:
                raise AttributeError('Source type ({}) not implemented'.format(source_type))

        sample_range = tqdm(range(num_samples)) if verbose is True else range(num_samples)

        if n_jobs == 1:
            pixels = [core.run_symmetric(
                np.array(self.coefficients.attrs['diffusion']['tensor_ratio'], dtype=np.float64, order='C'),
                evolution_length, np.array(self.coefficients.principle_angle, dtype=np.float64, order='C'),
                np.array(self.v, dtype=np.float64, order='C'),
                np.array(self.coefficients.diffusion_coefficient, dtype=np.float64, order='C'),
                np.array(self.coefficients.correlation_time, dtype=np.float64, order='C'),
                np.array(source[i], dtype=np.float64, order='C'), verbose) for i in sample_range]
        else:
            pixels = Parallel(n_jobs=min(n_jobs, num_samples))(
                delayed(core.run_symmetric)(
                np.array(self.coefficients.attrs['diffusion']['tensor_ratio'], dtype=np.float64, order='C'),
                evolution_length, np.array(self.coefficients.principle_angle, dtype=np.float64, order='C'),
                np.array(self.v, dtype=np.float64, order='C'),
                np.array(self.coefficients.diffusion_coefficient, dtype=np.float64, order='C'),
                np.array(self.coefficients.correlation_time, dtype=np.float64, order='C'),
                np.array(source[i], dtype=np.float64, order='C'), verbose) for i in sample_range)

        movie = xr.DataArray(data=pixels,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, self.num_frames),
                                     'x': self.coefficients.x, 'y': self.coefficients.y},
                             dims=['sample', 't', 'x', 'y'])

        movie.attrs['source'] = source_attr
        movie.attrs['seed'] = self.seed if source_attr=='random (numpy)' else None
        if num_samples == 1:
            movie = movie.squeeze('sample')

        return movie

    def get_laplacian(self, movie):
        lap = core.get_laplacian(
            self.coefficients.diffusion['tensor_ratio'],
            np.array(self.coefficients.principle_angle, dtype=np.float64, order='C'),
            np.array(self.coefficients.diffusion_coefficient, dtype=np.float64, order='C'),
            np.array(self.v, dtype=np.float64, order='C'),
            np.array(self.coefficients.correlation_time, dtype=np.float64, order='C'),
            np.array(movie, dtype=np.float64, order='C')
        )
        laplacian = xr.DataArray(data=lap, coords=movie.coords, dims=movie.dims)
        return laplacian

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def num_frames(self):
        return self.coefficients.num_frames

    @property
    def forcing_strength(self):
        return self.coefficients.forcing_strength

    @property
    def diffusion(self):
        diffusion = self.coefficients[self._diffusion_vars]
        diffusion.attrs = self.coefficients.diffusion
        return diffusion

    @property
    def advection(self):
        advection = self.coefficients[self._advection_vars]
        advection.attrs = self.coefficients.advection
        return advection

    @property
    def v(self):
        return np.stack([self.coefficients.vx, self.coefficients.vy], axis=-1)

    @property
    def pixels(self):
        return self.coefficients.pixels

    @property
    def seed(self):
        return self.coefficients.seed

