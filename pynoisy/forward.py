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

    def __init__(self, advection, diffusion, forcing_strength=1.0, seed=None):
        self._coefficients = xr.merge([advection, diffusion])
        self._coefficients.attrs['advection'] = advection.attrs
        self._coefficients.attrs['diffusion'] = diffusion.attrs
        self._coefficients.attrs['forcing_strength'] = forcing_strength
        self.reseed(seed)
        self._coefficients.attrs['num_frames'] = core.get_num_frames()
        self._diffusion_vars = list(diffusion.data_vars.keys())
        self._advection_vars = list(advection.data_vars.keys())

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
        source = np.random.randn(num_samples, self.num_frames, *core.get_image_size()) * self.forcing_strength \
            if source is None else source

        pixels = core.run_symmetric(
            np.array(self.coefficients.attrs['diffusion']['tensor_ratio'], dtype=np.float64, order='C'),
            evolution_length, np.array(self.coefficients.principle_angle),
            np.array(self.v, dtype=np.float64, order='C'),
            np.array(self.coefficients.diffusion_coefficient, dtype=np.float64, order='C'),
            np.array(self.coefficients.correlation_time, dtype=np.float64, order='C'),
            np.array(source, dtype=np.float64, order='C'), verbose
        )
        movie = xr.DataArray(data=pixels,
                             coords={'t': np.linspace(0, evolution_length, self.num_frames),
                                     'x': self.coefficients.x, 'y': self.coefficients.y},
                             dims=['t', 'x', 'y'])
        return movie

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def num_frames(self):
        return self.coefficients.attrs['num_frames']

    @property
    def forcing_strength(self):
        return self.coefficients.attrs['forcing_strength']

    @property
    def diffusion(self):
        diffusion = self.coefficients[self._diffusion_vars]
        diffusion.attrs = self.coefficients.attrs['diffusion']
        return diffusion

    @property
    def advection(self):
        advection = self.coefficients[self._advection_vars]
        advection.attrs = self.coefficients.attrs['advection']
        return advection

    @property
    def v(self):
        return np.stack([self.coefficients.vx, self.coefficients.vy], axis=-1)

    @property
    def pixels(self):
        return self.coefficients.pixels

    @property
    def seed(self):
        return self.coefficients.attrs['seed']

