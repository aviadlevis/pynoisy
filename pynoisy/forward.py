"""
TODO: Some documentation and general description goes here.
"""
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import core
from pynoisy.advection import Advection
from pynoisy.diffusion import Diffusion
from pynoisy import Movie, MovieSamples


class PDESolver(object):
    """TODO"""
    def __init__(self, advection=Advection(), diffusion=Diffusion(), forcing_strength=1.0, seed=None):
        self._forcing_strength = forcing_strength
        self.set_advection(advection)
        self.set_diffusion(diffusion)
        self.reseed(seed)
        self._num_frames = core.get_num_frames()

    def reseed(self, seed=None):
        self._seed = np.random.randint(0, 32767) if seed is None else seed
        print('Setting solver seed to: {}'.format(self.seed), end='\r')

    def set_advection(self, advection):
        """TODO"""
        self._advection = advection

    def set_diffusion(self, diffusion):
        """TODO"""
        self._diffusion = diffusion

    def run(self, evolution_length=0.1, verbose=True, myid=0):
        """TODO"""
        frames = core.run_main(
            self.diffusion.tensor_ratio,
            self.forcing_strength,
            evolution_length,
            self.diffusion.principle_angle,
            self.advection.v,
            self.diffusion.diffusion_coefficient,
            self.diffusion.correlation_time,
            verbose,
            self.seed + myid
        )
        return Movie(frames, duration=evolution_length)

    def run_adjoint(self, source, evolution_length=0.1, verbose=True):
        """TODO"""
        frames = core.run_adjoint(
            self.diffusion.tensor_ratio,
            evolution_length,
            self.diffusion.principle_angle,
            self.advection.v,
            self.diffusion.diffusion_coefficient,
            self.diffusion.correlation_time,
            np.array(source, dtype=np.float64, order='C'),
            verbose
        )
        return Movie(frames, duration=evolution_length)

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def forcing_strength(self):
        return self._forcing_strength

    @property
    def advection(self):
        return self._advection

    @property
    def diffusion(self):
        return self._diffusion

    @property
    def seed(self):
        return self._seed

class SamplerPDESolver(PDESolver):
    def __init__(self, advection=Advection(), diffusion=Diffusion(), forcing_strength=1.0, num_samples=1, seed=None):
        super().__init__(advection, diffusion, forcing_strength, seed)
        self._num_samples = num_samples

    def run(self, evolution_length=0.1, n_jobs=1, verbose=True):
        sample_range = tqdm(range(self.num_samples)) if verbose is True else range(self.num_samples)
        if n_jobs == 1:
            movie_list = [super(SamplerPDESolver, self).run(evolution_length, verbose=False) for i in sample_range]
        else:
            movie_list = Parallel(n_jobs=min(n_jobs, self.num_samples))(
                delayed(super(SamplerPDESolver, self).run)(evolution_length, verbose=False, myid=i) for i in sample_range)
        return MovieSamples(movie_list)

    @property
    def num_samples(self):
        return self._num_samples