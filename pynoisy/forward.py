"""
TODO: Some documentation and general description goes here.
"""
import xarray as xr
import numpy as np
from joblib import Parallel, delayed
import warnings
import subprocess
import os, tempfile, time
from scipy.special import gamma
from os import getpid

class HGRFSolver(object):
    """TODO"""

    solver_list = ['PCG', 'SMG']
    def __init__(self, advection, diffusion, forcing_strength=1.0, seed=None, num_solvers=1,
                 solver_type='PCG', executable='matrices'):

        # Check input parameters
        if not(np.allclose(advection.coords['x'] , diffusion.coords['x']) and \
                np.allclose(advection.coords['y'] , diffusion.coords['y'])):
            raise NotImplementedError('different coordinates for diffusion and advection not implemented')
        if solver_type not in self.solver_list:
            raise AttributeError('Not supported solver type: {}'.format(solver_type))

        self.advection = advection
        self.diffusion = diffusion
        self.params = xr.Dataset({
            'seed': seed,
            'forcing_strength': forcing_strength,
            'solver_type': solver_type,
            'solver_id': self.solver_list.index(solver_type),
            'executable': executable,
            'num_solvers': num_solvers
        })
        if seed is None:
            self.reseed()

        self._param_file = tempfile.NamedTemporaryFile(suffix='.h5')
        self._src_file = tempfile.NamedTemporaryFile(suffix='.h5') if num_solvers == 1 else \
            [tempfile.NamedTemporaryFile(suffix='.h5') for _ in range(num_solvers)]
        self._output_file = tempfile.NamedTemporaryFile(suffix='.h5') if num_solvers == 1 else \
            [tempfile.NamedTemporaryFile(suffix='.h5') for _ in range(num_solvers)]

    def __del__(self):
        del self._param_file, self._src_file, self._output_file

    def reseed(self, seed=None):
        np.random.seed(hash(time.time()) % 4294967295)
        self.params['seed'] = np.random.randint(0, 32767) if seed is None else seed
        print('Setting solver seed to: {}'.format(self.seed))

    def update_advection(self, advection):
        """TODO"""
        self.advection = advection
        self.advection.attrs.update(advection.attrs)

    def update_diffusion(self, diffusion):
        """TODO"""
        self.diffusion = diffusion
        self.diffusion.attrs.update(diffusion.attrs)

    def sample_source(self, nt, evolution_length=100.0, num_samples=1, seed=None):
        seed = self.seed if seed is None else seed
        np.random.seed(seed)
        source = xr.DataArray(
            data=np.random.randn(num_samples, nt, self.nx, self.ny) * self.forcing_strength,
            coords={'sample': range(num_samples),  't': np.linspace(0, evolution_length, nt), 'x': self.x, 'y': self.y},
            dims=['sample', 't', 'x', 'y'],
            attrs = {'seed': seed,
                     't_units': 't in terms of M: t = G M / c^3 for units s.t. G = 1 and c = 1.'}
        )
        return source

    def save(self, path):
        self.params.to_netcdf(path, mode='w')
        self.advection.to_netcdf(path, group='advection', mode='a')
        self.diffusion.to_netcdf(path, group='diffusion', mode='a')

    def run(self, source=None, nt=None, evolution_length=100.0, maxiter=50, nrecur=1, verbose=2, num_samples=1, tol=1e-6,
            n_jobs=1, seed=None, timer=False, std_scaling=True, nprocx=1, nprocy=1, nproct=-1):
        """TODO"""

        # Check input parameters
        seed = self.seed if seed is None else seed
        if (source is None) and (nt is None):
            raise AttributeError('If the source is unspecified, the number of frames (nt) should be specified.')

        if source is None:
            source = self.sample_source(nt, evolution_length, num_samples, seed)
        else:
            nt = source.sizes['t']
            evolution_length = source.t.max().data

        if not np.log2(nt).is_integer():
            warnings.warn("Warning: number of frames is not a power(2), this is suboptimal")

        self.save(self._param_file.name)

        n_jobs, proccesing_cmd = self._parallel_processing_cmd(nt, n_jobs, nprocx, nprocy, nproct)
        cmd = ['mpiexec', '-n', str(n_jobs), str(self.params.executable.data), '-seed', str(seed),  '-tol', str(tol),
               '-maxiter', str(maxiter), '-verbose', str(int(verbose)), '-nrecur', str(nrecur),
               '-dump', '-params', self._param_file.name, '-solver', str(self.params.solver_id), '-tend',
               str(evolution_length), '-x1start', str(self.x[0].data), '-x2start',  str(self.y[0].data),
               '-x1end',  str(self.x[-1].data), '-x2end', str(self.y[-1].data)]
        cmd.extend(proccesing_cmd)

        if timer is True:
            cmd.append('-timer')

        if std_scaling:
            attrs = source.attrs
            source = source * self.std_scaling_factor(nrecur)
            source.attrs.update(attrs)

        # Parallel processing
        if self.params.num_solvers == 1:
            output_dir = os.path.dirname(self._output_file.name)
            output_filename = os.path.relpath(self._output_file.name, output_dir)
            cmd.extend(['-source', self._src_file.name, '-output', output_dir, '-filename', output_filename])
            output = []
            for sample in range(source.sample.size):
                source.sel(sample=sample).to_dataset(name='data_raw').to_netcdf(self._src_file.name, group='data', mode='w')
                subprocess.run(cmd)
                output.append((sample, xr.load_dataset(self._output_file.name, group='data')))

        elif self.params.num_solvers > 1:
            output = Parallel(n_jobs=self.params.num_solvers)(
                delayed(self._parallel_run)(
                    cmd, source.isel(sample=sample), sample,
                    input_names=[file.name for file in self._src_file],
                    output_names=[file.name for file in self._output_file],
                    n_jobs=self.params.num_solvers)
                for sample in range(source.sample.size)
            )

        dims = ['t', 'x', 'y']
        coords = {'t': np.linspace(0, evolution_length, nt), 'x': self.x, 'y': self.y}
        attrs = {'tol': tol, 'maxiter': maxiter, 'solver_type': str(self.solver_type.data),
                 'std_scaling': str(std_scaling)}
        attrs.update(source.attrs)

        for sample, pixels in output:
            movies = [
                xr.DataArray(pixels['step_{}'.format(deg)].data, coords=coords, dims=dims, attrs=attrs).assign_coords(
                    deg=deg + 1, sample=sample) for deg in range(nrecur - 1)
            ]
            final_movie = xr.DataArray(data=pixels.data_raw.data, coords=coords, dims=dims, attrs=attrs)
            final_movie = final_movie.assign_coords(deg=nrecur, sample=sample)
            movies.append(final_movie)
            output[sample] = xr.concat(movies, dim='deg')
        output = xr.concat(output, dim='sample')

        if (nrecur == 1):
            output = output.squeeze('deg').drop_vars('deg')

        if (output.sample.size == 1):
            output = output.squeeze('sample').drop_vars('sample')

        return output

    def std_scaling_factor(self, nrecur=1, threshold=1e-10):
        factor = (4. * np.pi) ** (3. / 2.) * gamma(2. * nrecur) / gamma(2. * nrecur - 3. / 2.)
        return np.sqrt(factor * self.diffusion.tensor_ratio * self.diffusion.correlation_time *
                       self.diffusion.correlation_length ** 2).clip(min=threshold)

    def _parallel_run(cmd, source, sample, input_names, output_names, n_jobs):
        file_id = np.mod(getpid(), n_jobs)
        output_dir = os.path.dirname(output_names[file_id])
        output_filename = os.path.relpath(output_names[file_id], output_dir)
        cmd.extend(['-source', input_names[file_id], '-output', output_dir, '-filename', output_filename])
        source.to_dataset(name='data_raw').to_netcdf(input_names[file_id], group='data', mode='w')
        subprocess.run(cmd)
        output = xr.load_dataset(output_names[file_id], group='data')
        return (sample, output)

    def _parallel_processing_cmd(self, nt, n_jobs, nprocx, nprocy, nproct):
        """
        Determine the domain distribution among processors and output the command line arguments.
        """
        assert np.mod(self.nx, nprocx) == 0, 'nx / nprocx should be an integer'
        assert np.mod(self.ny, nprocy) == 0, 'ny / nprocy should be an integer'
        assert np.mod(nt, nproct) == 0, 'nt / nproct should be an integer'

        if (nprocx == -1):
            assert (nprocy == 1) and (nproct == 1), 'nprocx is set to -1 with nprocy or nproct different than 1.'
            nprocx = n_jobs
        elif (nprocy == -1):
            assert (nprocx == 1) and (nproct == 1), 'nprocy is set to -1 with nprocx or nproct different than 1.'
            nprocy = n_jobs
        elif (nproct == -1):
            assert (nprocx == 1) and (nprocy == 1), 'nproct is set to -1 with nprocx or nprocy different than 1.'
            nproct = n_jobs
        elif (nprocx*nprocy*nproct != n_jobs):
            n_jobs = nprocx*nprocy*nproct
            warnings.warn('n_jobs is overwritten by {}: n_jobs != nprocx*nprocy*nproct.'.format(n_jobs))

        cmd = [
            '-ni', str(int(self.nx / nprocx)), '-nj', str(int(self.ny / nprocy)), '-nk', str(int(nt / nproct)),
            '-pgrid', str(int(nprocx)), str(int(nprocy)), str(int(nproct))
        ]
        return n_jobs, cmd


    @classmethod
    def from_netcdf(cls, path):
        params = xr.load_dataset(path)
        return cls(advection=xr.load_dataset(path, group='advection'),
                   diffusion=xr.load_dataset(path, group='diffusion'),
                   forcing_strength=params.forcing_strength.data,
                   seed=params.seed.data,
                   num_solvers=params.num_solvers,
                   solver_type=params.solver_type,
                   executable=params.executable)

    @property
    def nx(self):
        return self.advection.x.size

    @property
    def ny(self):
        return self.advection.y.size

    @property
    def x(self):
        return self.advection.x

    @property
    def y(self):
        return self.advection.y

    @property
    def seed(self):
        return int(self.params.seed)

    @property
    def solver_type(self):
        return self.params['solver_type']

    @property
    def forcing_strength(self):
        return float(self.params['forcing_strength'])

    @property
    def v(self):
        return np.stack([self.advection.vx, self.advection.vy], axis=-1)


