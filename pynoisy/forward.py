"""
Forward generation classes and functions to compute solutions to the stochastic partial differential equation (SPDE)
ofr anistropic spatio-temporal-diffusion [1].

References
----------
.. [1] Lee, D. and Gammie, C.F., 2021. Disks as Inhomogeneous, Anisotropic Gaussian Random Fields.
    The Astrophysical Journal, 906(1), p.39.
    url: https://iopscience.iop.org/article/10.3847/1538-4357/abc8f3/meta
.. [2] url: https://github.com/hypre-space/hypre
"""
import xarray as _xr
import numpy as _np
from joblib import Parallel as _Parallel, delayed as _delayed
import warnings as _warnings
import subprocess as _subprocess
import os as _os
import tempfile as _tempfile
import time as _time
from scipy.special import gamma as _gamma

class HGRFSolver(object):
    solver_list = ['PCG', 'SMG', 'Inverse']
    def __init__(self, advection, diffusion, nt, evolution_length=100.0,  t_units='GM/c^3', forcing_strength=1.0, seed=None, num_solvers=1,
                 solver_type='PCG', executable='matrices'):
        """
        The Hypre Gaussian Random Field (HGRF) solver contains all the diffusion tensor fields [1] and facilitates
        input/output writing to pass parameters to and from HYPRE [2]. Input parameters and output GRFs are written to
        files using temporary .h5 files (with tempfile package).

        Parameters
        ----------
        advection: xr.Dataset
            A Dataset specifying the advection velocities 'vx' and 'vy' on a 2D grid. (see pynoisy/advection.py)
        diffusion: xr.Dataset
            A Dataset specifying the diffusion fields on a 2D grid (see pynoisy/diffusion.py):
                - correlation_length (2D field)
                - correlation_time (2D field)
                - spatial_angle (2D field)
                - tensor_radio (scalar).
        nt: int,
            Number of temporal frames (should be a power of 2).
        evolution_length: float, default=100.
            Evolution time in terms of units
        t_units: str, default='GM/c^3'
            t in terms of M: t = G M / c^3 for units s.t. G = 1 and c = 1.
        forcing_strength: float, default=1.0,
            The standard deviation of the random source.
        seed: int, optional,
            Random seed for numpy.random. If not specified a random seed is drawn.
        num_solvers: int, default=1,
            Number of *parallel* solvers.
        solver_type: string, default='PCG',
            'PCG' or 'SMG' solver types are supported for use within HYPRE Struct interface (see [3,4] for more info)
        executable: string, default='matrices',
            Currently only matrices executable is supported.



        References
        ----------
        .. [1] Lee, D. and Gammie, C.F., 2021. Disks as Inhomogeneous, Anisotropic Gaussian Random Fields.
            The Astrophysical Journal, 906(1), p.39.
            url: https://iopscience.iop.org/article/10.3847/1538-4357/abc8f3/meta
        .. [2] url: https://github.com/hypre-space/hypre
        .. [3] url: https://hypre.readthedocs.io/en/latest/ch-struct.html
        .. [4] url: https://hypre.readthedocs.io/en/latest/ch-solvers.html
        """
        # Check input parameters
        if not(_np.allclose(advection.coords['x'] , diffusion.coords['x']) and \
               _np.allclose(advection.coords['y'] , diffusion.coords['y'])):
            raise NotImplementedError('different coordinates for diffusion and advection not implemented')
        if solver_type not in self.solver_list:
            raise AttributeError('Not supported solver type: {}'.format(solver_type))
        if not _np.log2(nt).is_integer():
            _warnings.warn("Warning: number of frames is not a power(2), this is suboptimal")

        self.advection = advection
        self.diffusion = diffusion
        self.params = _xr.Dataset({
            'nt': nt,
            'evolution_length': evolution_length,
            't_units': t_units,
            'seed': seed,
            'forcing_strength': forcing_strength,
            'solver_type': solver_type,
            'solver_id': self.solver_list.index(solver_type),
            'executable': executable,
            'num_solvers': num_solvers
        })
        if seed is None:
            self.reseed(printval=False)

        self._param_file = _tempfile.NamedTemporaryFile(suffix='.h5')
        self._src_file = _tempfile.NamedTemporaryFile(suffix='.h5') if num_solvers == 1 else \
            [_tempfile.NamedTemporaryFile(suffix='.h5') for _ in range(num_solvers)]
        self._output_file = _tempfile.NamedTemporaryFile(suffix='.h5') if num_solvers == 1 else \
            [_tempfile.NamedTemporaryFile(suffix='.h5') for _ in range(num_solvers)]

    def __del__(self):
        """
        Delete temporary files.
        """
        del self._param_file, self._src_file, self._output_file

    def reseed(self, seed=None, printval=True):
        """
        Reseed the random number generator.

        Parameters
        ----------
        seed: int, optional,
            If None, a random seed is drawn according to the datetime.
        printval: bool, default=True
            Print new seed value

        Notes
        -----
        The seed is stored in self.params['seed'].
        """
        _np.random.seed(hash(_time.time()) % 4294967295)
        self.params['seed'] = _np.random.randint(0, 32767) if seed is None else seed
        if printval:
            print('Setting solver seed to: {}'.format(self.seed))

    def update_advection(self, advection):
        """
        Update the solver's advection fields.

        Parameters
        ----------
        advection: xr.Dataset
            A Dataset specifying the advection velocities 'vx' and 'vy' on a 2D grid. (see pynoisy/advection.py)
        """
        self.advection = advection

    def update_diffusion(self, diffusion):
        """
        Update the solver's diffusion fields.

        Parameters
        ----------
        diffusion: xr.Dataset
            A Dataset specifying the diffusion fields on a 2D grid (see pynoisy/diffusion.py):
                - correlation_length (2D field)
                - correlation_time (2D field)
                - spatial_angle (2D field)
                - tensor_radio (scalar) .
        """
        self.diffusion = diffusion

    def sample_source(self, num_samples=1, seed=None):
        """
        Sample a random Gaussian source.

        Parameters
        ----------
        num_samples: int, default=1,
            Number of random source samples
        seed: int, optional,
            If None, the seed stored in self.seed is used.

        Returns
        -------
        source: xr.DataArray,
            Random Gaussian source DataArray with dimensions ['sample', 't', 'y', 'x']
        """
        seed = self.seed if seed is None else seed
        _np.random.seed(seed)
        source = _xr.DataArray(
            data=_np.random.randn(num_samples, self.nt, self.ny, self.nx) * self.forcing_strength,
            coords={'sample': range(num_samples),  't': self.t, 'y': self.y, 'x': self.x},
            dims=['sample', 't', 'y', 'x'],
            attrs = {'seed': seed}
        )
        return source

    def to_netcdf(self, path, group='', mode='w'):
        """
        Save solver parameters (params, advection, diffusion) to netcdf.
        Parameters are saved to the same file with different groups.

        Parameters
        ----------
        path: str,
            Output file path.
        group: str, default=''
            Group to nest solver within
        mode: ({"w", "a"}, default: "w"
            Write (‘w’) or append (‘a’) mode. If mode=’w’, any existing file at this location will be overwritten.
            If mode=’a’, existing variables will be overwritten.

        Notes
        -----
        For loading see class method from_netcdf.
        """
        self.params.to_netcdf(path, group=group + 'params', mode=mode)
        self.advection.to_netcdf(path, group=group + 'advection', mode='a')
        self.diffusion.to_netcdf(path, group=group + 'diffusion', mode='a')

    def run(self, source=None, maxiter=50, nrecur=1, verbose=2, num_samples=1, tol=1e-6,
            n_jobs=1, seed=None, timer=False, std_scaling=True, nprocx=1, nprocy=1, nproct=-1):
        """
        Run the SPDE solver to sample a Gaussian Random Field (GRF).

        Parameters
        ----------
        source: xr.DataArray, optional,
            A user specified source. The source DataArray has dims ['sample', 't', 'y', 'x'] ('sample' is optional).
            If source is not specified, a random gaussian source is used (see HGRFSolver.sample_source method)
        maxiter: int, default=50,
            Maximum number of iteration of the underlying HYPRE (PCG or SMG) solver.
        nrecur: int, default=1,
            Number of recursive computations.
        verbose: int, default=2,
            Level of verbosity (1-2).
        num_samples: int, default=1,
            Number of GRF samples. Only used if source is unspecified.
        tol: float, default=1e-6,
            tolerance for the underlying HYPRE (PCG or SMG) solver iterations.
        n_jobs: int, default=1,
            Number of MPI jobs. MPI is used to subdivide the medium according to nprocx/nprocy/nproct.
        seed: int, optional,
            If None, the seed stored in self.seed is used.
        timer: bool, default=False,
            Output timing of intermediate computations.
        std_scaling: bool, default=True,
            Scale the input source according to the determinant of the diffusion tensor.
            This yields flat (constant) temporal variance across image pixels.
        nprocx: int, default=1,
            Number of MPI processes in x dimension.
        nprocy: int, default=1,
            Number of MPI processes in y dimension.
        nproct: int, default=-1,
            Number of MPI processes in t dimension. Default -1 is using all of n_jobs in t dimension.

        Returns
        -------
        grf: xr.DataArray
            Output DataArray GRF with dims ['sample', 't', 'y', 'x'] ('sample' is optional).

        Notes
        -----
        For nprocs>0 the product should equal the number of jobs: nprox*nproxy*nproxt = n_jobs.
        """
        seed = self.seed if seed is None else seed
        if source is None:
            source = self.sample_source(num_samples, seed)
        else:
            if (source.sizes['t'] != self.nt):
                raise AttributeError('Input source has different number of frames to solver: {} != {}.'.format(
                    source.sizes['t'], self.nt))

        # Save parameters to file (loaded within HYPRE)
        self.to_netcdf(self._param_file.name)
        n_jobs, proccesing_cmd = self._parallel_processing_cmd(n_jobs, nprocx, nprocy, nproct)
        cmd = ['mpiexec', '-n', str(n_jobs), self.executable, '-seed', str(seed),  '-tol', str(tol),
               '-maxiter', str(maxiter), '-verbose', str(int(verbose)), '-nrecur', str(nrecur),
               '-dump', '-params', self._param_file.name, '-solver', str(self.params.solver_id.data), '-tend',
               str(self.evolution_length), '-x1start', str(self.x[0].data), '-x2start',  str(self.y[0].data),
               '-x1end',  str(self.x[-1].data), '-x2end', str(self.y[-1].data)]
        cmd.extend(proccesing_cmd)

        if timer is True:
            cmd.append('-timer')

        # Scale source according to diffusion tensor determinant
        if std_scaling:
            attrs = source.attrs
            source = source * self.std_scaling_factor(nrecur)
            source.attrs.update(attrs)

        # Transpose source x,y coordinates for HYPRE (k, j, i) = (t, x, y)
        source = source.expand_dims('sample') if 'sample' not in source.dims else source
        source = source.transpose('sample', 't', 'x', 'y')

        if self.num_solvers == 1:
            output_dir = _os.path.dirname(self._output_file.name)
            output_filename = _os.path.relpath(self._output_file.name, output_dir)
            cmd.extend(['-source', self._src_file.name, '-output', output_dir, '-filename', output_filename])
            output = []
            for sample in range(source.sample.size):
                source.sel(sample=sample).to_dataset(name='data_raw').to_netcdf(self._src_file.name, group='data', mode='w')
                _subprocess.run(cmd)
                output.append((sample, _xr.load_dataset(self._output_file.name, group='data')))

        # Parallel processing: this is a parallelization layer on top of the MPI.
        # i.e. multiple solvers each splitting the medium according to n_jobs
        elif self.num_solvers > 1:

            def _parallel_run(cmd, source, sample, input_names, output_names, n_jobs):
                file_id = _np.mod(_os.getpid(), n_jobs)
                output_dir = _os.path.dirname(output_names[file_id])
                output_filename = _os.path.relpath(output_names[file_id], output_dir)
                cmd.extend(['-source', input_names[file_id], '-output', output_dir, '-filename', output_filename])
                source.to_dataset(name='data_raw').to_netcdf(input_names[file_id], group='data', mode='w')
                _subprocess.run(cmd)
                output = _xr.load_dataset(output_names[file_id], group='data')
                return (sample, output)

            output = _Parallel(n_jobs=int(self.num_solvers))(
                _delayed(_parallel_run)(
                    cmd, source.isel(sample=sample), sample,
                    input_names=[file.name for file in self._src_file],
                    output_names=[file.name for file in self._output_file],
                    n_jobs=int(self.num_solvers))
                for sample in range(source.sample.size)
            )

        dims = ['t', 'x', 'y']

        coords = {'t': source.t, 'x': self.x, 'y': self.y}
        attrs = {
            'forcing_strength': self.forcing_strength,
            'tol': tol,
            'maxiter': maxiter,
            'solver_type': self.solver_type,
            'std_scaling': str(std_scaling)
        }
        attrs.update(source.attrs)

        for sample, pixels in output:
            movies = [
                _xr.DataArray(pixels['step_{}'.format(deg)].data, coords=coords, dims=dims, attrs=attrs).assign_coords(
                    deg=deg + 1, sample=sample) for deg in range(nrecur - 1)
            ]
            final_movie = _xr.DataArray(data=pixels.data_raw.data, coords=coords, dims=dims, attrs=attrs)
            final_movie = final_movie.assign_coords(deg=nrecur, sample=sample)
            movies.append(final_movie)
            output[sample] = _xr.concat(movies, dim='deg')
        output = _xr.concat(output, dim='sample')

        if (nrecur == 1):
            output = output.squeeze('deg').drop_vars('deg')

        if (output.sample.size == 1):
            output = output.squeeze('sample').drop_vars('sample')

        grf = output.transpose(...,'t','y','x')
        grf.name = 'grf'
        return grf

    def run_inverse(self, source):
        """
        Run the "inverse" operation: sparse matrix (differential operators) multiplication

        Parameters
        ----------
        source: xr.DataArray,
            A user specified source. The source DataArray has dims ['sample', 't', 'y', 'x'] ('sample' is optional).
            If source is not specified, a random gaussian source is used (see HGRFSolver.sample_source method)

        Returns
        -------
        output: xr.DataArray
            Output DataArray GRF with dims [..., 't', 'y', 'x'].
        """
        inverse_solver = HGRFSolver(self.advection, self.diffusion, self.nt, self.evolution_length, self.t_units,
                                    self.forcing_strength, self.seed, self.num_solvers, 'Inverse', self.executable)
        output = inverse_solver.run(source=source, std_scaling=False) / \
                (self.forcing_strength * inverse_solver.std_scaling_factor())
        return output

    def std_scaling_factor(self, nrecur=1, threshold=1e-10):
        """
        Scale factor according to the determinant of the diffusion tensor.
        Multiplying the random source by this factor yields flat (constant) temporal variance across image pixels.

        Parameters
        ----------
        nrecur: int, default=1,
            Number of recursive computations.
        threshold: float, default=1e-10
            Clip the scale factor to a minimal value.
        """
        factor = (4. * _np.pi) ** (3. / 2.) * _gamma(2. * nrecur) / _gamma(2. * nrecur - 3. / 2.)
        return _np.sqrt(factor * self.diffusion.tensor_ratio * self.diffusion.correlation_time *
                        self.diffusion.correlation_length ** 2).clip(min=threshold)

    def _parallel_processing_cmd(self, n_jobs, nprocx, nprocy, nproct):
        """
        Determine the domain distribution among processors and output the command line arguments.

        Parameters
        ----------
        n_jobs: int, default=1,
            Number of MPI jobs. MPI is used to subdivide the medium according to nprocx/nprocy/nproct.
        nprocx: int, default=1,
            Number of MPI processes in x dimension.
        nprocy: int, default=1,
            Number of MPI processes in y dimension.
        nproct: int, default=-1,
            Number of MPI processes in t dimension. Default -1 is using all of n_jobs in t dimension.

        Notes
        -----
        For positive nprocs the product should equal the number of jobs: nprox*nproxy*nproxt = n_jobs.
        """
        assert _np.mod(self.nx, nprocx) == 0, 'nx / nprocx should be an integer'
        assert _np.mod(self.ny, nprocy) == 0, 'ny / nprocy should be an integer'
        assert _np.mod(self.nt, nproct) == 0, 'nt / nproct should be an integer'

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
            _warnings.warn('n_jobs is overwritten by {}: n_jobs != nprocx*nprocy*nproct.'.format(n_jobs))

        cmd = [
            '-ni', str(int(self.ny / nprocy)), '-nj', str(int(self.nx / nprocx)), '-nk',
            str(int(self.nt / nproct)), '-pgrid', str(int(nprocy)), str(int(nprocx)), str(int(nproct))
        ]
        return n_jobs, cmd

    @classmethod
    def from_netcdf(cls, path, group=''):
        """
        Load solver from netcdf file.

        Parameters
        ----------
        path: str,
            Output file path.
        group: str, default=''
            Group to nest solver within

        Returns
        -------
        solver: pynoisy.forward.HGRFSolver,
            A Solver object initialized from filed with saved parameters (params, advection, diffusion).

        Notes
        -----
        For saving see method to_netcdf.
        """
        params = _xr.load_dataset(path, group=group + 'params')
        return cls(advection=_xr.load_dataset(path, group=group + 'advection'),
                   diffusion=_xr.load_dataset(path, group=group + 'diffusion'),
                   nt=int(params.nt.data),
                   evolution_length=float(params.evolution_length.data),
                   t_units=str(params.t_units.data),
                   forcing_strength=float(params.forcing_strength.data),
                   seed=int(params.seed.data),
                   num_solvers=int(params.num_solvers),
                   solver_type=str(params.solver_type.data),
                   executable=str(params.executable.data))

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
    def t(self):
        return _xr.DataArray(_np.arange(0, self.evolution_length, self.evolution_length/self.nt),
                             dims='t', attrs={'units': self.t_units})

    @property
    def nt(self):
        return int(self.params['nt'])

    @property
    def evolution_length(self):
        return float(self.params['evolution_length'])

    @property
    def t_units(self):
        return str(self.params['t_units'].data)

    @property
    def seed(self):
        return int(self.params['seed'])

    @property
    def solver_type(self):
        return str(self.params['solver_type'].data)

    @property
    def num_solvers(self):
        return int(self.params['num_solvers'])

    @property
    def executable(self):
        return str(self.params['executable'].data)

    @property
    def forcing_strength(self):
        return float(self.params['forcing_strength'])

    @property
    def v(self):
        return _np.stack([self.advection.vx, self.advection.vy], axis=-1)

def modulate(envelope, grf, alpha, keep_attrs=True):
    """
    Modulate an envelope by an exponential of the Gaussian Random Field (GRF):
        movie = envelope * np.exp(alpha * grf)

    Parameters
    ----------
    envelope: xr.DataArray,
        An image DataArray with dimensions ['y', 'x'].
    grf: xr.DataArray
        Output DataArray GRF with dims [..., 't', 'y', 'x'].
    alpha: float, default=1.0
        Exponential rate
    keep_attrs: bool, default=True,
        Keep attributed of envelope and grf.

    Returns
    -------
    movie: xr.DataArray
        Output modulated movie DataArray with dims [..., 't', 'y', 'x'].

    Notes
    -----
    Envelope and GRF image coordinates need to be identical.
    """
    _np.testing.assert_equal(envelope['x'].data, grf['x'].data)
    _np.testing.assert_equal(envelope['y'].data, grf['y'].data)
    movie = _np.exp(alpha * grf) * envelope
    movie.attrs.update(alpha=alpha)

    if keep_attrs:
        movie.attrs.update(grf.attrs)
        movie.attrs.update(envelope.attrs)

    movie.name = 'movie'
    return movie