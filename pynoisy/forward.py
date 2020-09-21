"""
TODO: Some documentation and general description goes here.
"""
import xarray as xr
import numpy as np
import noisy_core
import pynoisy.utils as utils
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
import subprocess
import os, tempfile
from scipy.special import gamma

class Solver(object):
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
        self.reseed(seed)
        warnings.resetwarnings()

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

    def sample_source(self, num_frames, evolution_length=0.1, num_samples=1, seed=None):
        seed = self.seed if seed is None else seed
        np.random.seed(seed)
        source = np.random.randn(num_samples, num_frames, self.nx, self.ny) * self.forcing_strength
        movie = xr.DataArray(data=source,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, num_frames),
                                     'x': self.params.x, 'y': self.params.y},
                             dims=['sample', 't', 'x', 'y'])
        return movie

    def save(self, path):
        self.params.to_netcdf(path, mode='w')
        self.advection.to_netcdf(path, group='advection', mode='a')
        self.diffusion.to_netcdf(path, group='diffusion', mode='a')

    @classmethod
    def from_netcdf(cls, path):
        params = xr.load_dataset(path)
        solver = cls(params.x.size, params.y.size, advection=xr.load_dataset(path, group='advection'),
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


class NoisySolver(Solver):
    def __init__(self, nx, ny, advection, diffusion, forcing_strength=1.0, seed=None):
        super().__init__(nx, ny, advection, diffusion, forcing_strength, seed)
        self._params.attrs = {'solver_type': 'Noisy'}

    def copy(self, deep=True):
        """TODO"""
        return NoisySolver(self.nx, self.ny,
            advection=self.advection.copy(deep=deep), diffusion=self.diffusion.copy(deep=deep),
            forcing_strength=self.forcing_strength, seed=self.seed)

    def run_asymmetric(self, num_frames, evolution_length=0.1, verbose=True, num_samples=1, n_jobs=1, seed=None):
        """TODO"""
        seed = self.seed if seed is None else seed
        sample_range = tqdm(range(num_samples)) if verbose is True else range(num_samples)
        if n_jobs == 1:
            pixels = [noisy_core.run_asymmetric(
                num_frames, self.nx, self.ny,
                self.diffusion.tensor_ratio,
                self.forcing_strength, evolution_length,
                np.array(self.diffusion.spatial_angle, dtype=np.float64, order='C'),
                np.array(self.v, dtype=np.float64, order='C'),
                np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                verbose, seed) for i in sample_range]

        else:
            pixels = Parallel(n_jobs=min(n_jobs, num_samples))(
                delayed(noisy_core.run_asymmetric)(
                    num_frames, self.nx, self.ny,
                    self.diffusion.tensor_ratio,
                    self.forcing_strength, evolution_length,
                    np.array(self.diffusion.spatial_angle, dtype=np.float64, order='C'),
                    np.array(self.v, dtype=np.float64, order='C'),
                    np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                    np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                    False, seed + i) for i in sample_range)

        movie = xr.DataArray(data=pixels,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, num_frames),
                                     'x': self.params.x, 'y': self.params.y},
                             dims=['sample', 't', 'x', 'y'])
        if num_samples == 1:
            movie.attrs['seed'] = seed
            movie = movie.squeeze('sample')
        else:
            movie.coords['seed'] = ('sample', seed + np.arange(num_samples))
        return movie


    def run_symmetric(self, source=None, evolution_length=0.1, verbose=True, num_frames=None, num_samples=1, n_jobs=1, seed=None):
        """TODO"""

        # Either pre-determined source or randomly sampled from Gaussian distribution
        if source is None:
            assert num_frames is not None, 'If the source is unspecified, the number of frames should be specified.'
            source = self.sample_source(num_frames, evolution_length, num_samples, seed)
        else:
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
                evolution_length, np.array(self.diffusion.spatial_angle, dtype=np.float64, order='C'),
                np.array(self.v, dtype=np.float64, order='C'),
                np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                np.array(source[i], dtype=np.float64, order='C'), verbose) for i in sample_range]
        else:
            pixels = Parallel(n_jobs=min(n_jobs, num_samples))(
                delayed(noisy_core.run_symmetric)(
                    num_frames, self.nx, self.ny,
                    np.array(self.diffusion.tensor_ratio, dtype=np.float64, order='C'),
                    evolution_length, np.array(self.diffusion.spatial_angle, dtype=np.float64, order='C'),
                    np.array(self.v, dtype=np.float64, order='C'),
                    np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
                    np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
                    np.array(source[i], dtype=np.float64, order='C'), verbose) for i in sample_range)

        movie = xr.DataArray(data=pixels,
                             coords={'sample': range(num_samples),
                                     't': np.linspace(0, evolution_length, num_frames),
                                     'x': self.params.x, 'y': self.params.y},
                             dims=['sample', 't', 'x', 'y'])

        movie.attrs['seed'] = self.seed
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
            np.array(self.diffusion.spatial_angle, dtype=np.float64, order='C'),
            np.array(self.diffusion.diffusion_coefficient, dtype=np.float64, order='C'),
            np.array(self.v, dtype=np.float64, order='C'),
            np.array(self.diffusion.correlation_time, dtype=np.float64, order='C'),
            np.array(movie, dtype=np.float64, order='C')
        )
        laplacian = xr.DataArray(data=lap, coords=movie.coords, dims=movie.dims)
        return laplacian


class HGRFSolver(Solver):
    solver_list = ['PCG', 'SMG']
    random_types = ['numpy', 'zigur']

    def __init__(self, nx, ny, advection, diffusion, forcing_strength=1.0, seed=None, solver_type='SMG', random_type='numpy', executable='matrices'):
        super().__init__(nx, ny, advection, diffusion, forcing_strength, seed)
        assert solver_type in self.solver_list, 'Not supported solver type: {}'.format(solver_type)
        assert random_type in self.random_types, 'Not supported random type: {}'.format(random_type)
        self._solver_id = self.solver_list.index(solver_type)
        self._params.update({'solver_type': solver_type, 'random_type': random_type})
        self._params.attrs = {'solver_type': 'HGRF', 'executable': executable}
        self._param_file = tempfile.NamedTemporaryFile(suffix='.h5')
        self._src_file = tempfile.NamedTemporaryFile(suffix='.h5')
        self._output_file = tempfile.NamedTemporaryFile(suffix='.h5')

    def __del__(self):
        del self._param_file, self._src_file, self._output_file

    def run(self, maxiter=50, nrecur=1, evolution_length=0.1, source=None, verbose=2, num_frames=None, num_samples=1,
            n_jobs=1, seed=None, solver_id=None, timer=False, std_scaling=True):
        """TODO"""
        seed = self.seed if seed is None else seed
        solver_id = self._solver_id if solver_id is None else solver_id

        # Either pre-determined source or randomly sampled from Gaussian distribution
        output = os.path.dirname(self._output_file.name)
        filename = os.path.relpath(self._output_file.name, output)
        self.save(self._param_file.name)
        cmd = ['mpiexec', '-n', str(n_jobs), self.params.executable, '-ni', str(self.nx), '-nj', str(self.ny),
               '-seed', str(seed), '-output', output, '-filename', filename, '-maxiter', str(maxiter),
               '-verbose', str(int(verbose)), '-nrecur', str(nrecur), '-dump',
               '-params', self._param_file.name, '-solver', str(solver_id)]
        if timer is True:
            cmd.append('-timer')

        assert ((num_frames is not None) or (source is not None)), \
            'If the source is unspecified, the number of frames should be specified.'

        if (num_frames is not None) and (not np.log2(num_frames).is_integer()):
            warnings.warn("Warning: number of frames is not a power(2), this is suboptimal")

        if source is None:
            if self.params.random_type == 'zigur':
                raise NotImplementedError
            else:
                source = self.sample_source(num_frames, evolution_length, num_samples, seed)
        else:
            num_frames = source.sizes['t']

        if std_scaling:
            factor = (4. * np.pi) ** (3. / 2.) * gamma(2. * nrecur) / gamma(2. * nrecur - 3. / 2.)
            scaling = np.sqrt(factor * self.diffusion.tensor_ratio * self.diffusion.correlation_time *
                              self.diffusion.correlation_length ** 2).clip(min=1e-10)
            source = source * scaling

        mode = 'a' if solver_id == 3 else 'w'
        source.to_dataset(name='data_raw').to_netcdf(self._src_file.name, group='data', mode=mode)
        cmd.extend(['-source', self._src_file.name])

        assert np.mod(num_frames, n_jobs) == 0, 'Num frames / n_jobs should be an integer'
        cmd.extend(['-nk', str(num_frames / n_jobs)])
        subprocess.run(cmd)

        pixels = xr.load_dataset(self._output_file.name, group='data')
        coords = {'t': np.linspace(0, evolution_length, num_frames), 'x': self.params.x, 'y': self.params.y}
        dims = ['t', 'x', 'y']
        attrs = {'seed': seed}

        final_movie = xr.DataArray(data=pixels.data_raw.data, coords=coords, dims=dims, attrs=attrs)
        final_movie = final_movie.assign_coords(deg=nrecur)

        movies = [
            xr.DataArray(pixels['step_{}'.format(deg)].data, coords=coords, dims=dims, attrs=attrs).assign_coords(
                deg=deg + 1)
            for deg in range(nrecur - 1)
        ]

        movies.append(final_movie)
        movie = xr.concat(movies, dim='deg')
        if nrecur == 1:
            movie = movie.squeeze('deg')

        if (num_samples > 1):
            raise NotImplementedError
            # coords['sample'] = range(num_samples)
            # dims = ['sample'] + dims
            # coords['seed'] = ('sample', seed + np.arange(num_samples))

        return movie

    def get_laplacian(self, movie, verbose=0, timer=False):
        return self.run(solver_id=2, source=movie, verbose=verbose, timer=timer)

    def get_spatial_angle_gradient(self, forward, adjoint, verbose=0, timer=False):
        adjoint.to_dataset(name='adjoint').to_netcdf(self._src_file.name, group='data', mode='w')
        output = self.run(solver_id=3, source=forward, verbose=verbose, timer=timer)
        scaling = self.nx * self.ny * output.sizes['t']
        gradient = xr.DataArray(data=output.sum('t').data.T, coords=self.params.coords) / scaling
        return gradient



