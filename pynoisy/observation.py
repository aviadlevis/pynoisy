"""
Observations classes and functions to compute EHT interferometric observations.
Here is where the interface with eht-imaging [1] library resides.

References
----------
.. [1] eht-imaging: https://github.com/achael/eht-imaging
"""
import xarray as _xr
import numpy as _np
import gc as _gc
import matplotlib.pyplot as _plt
import scipy.ndimage as _nd
import ehtim as _eh
import ehtim.const_def as _ehc
import pynoisy.utils as _utils

def plot_uv_coverage(obs, ax=None, fontsize=14, cmap='rainbow', add_conjugate=True, xlim=(-9.5, 9.5), ylim=(-9.5, 9.5),
                     shift_inital_time=True):
    """
    Plot the uv coverage as a function of observation time.
    x axis: East-West frequency
    y axis: North-South frequency

    Parameters
    ----------
    obs: ehtim.Obsdata
        ehtim Observation object
    ax: matplotlib axis,
        A matplotlib axis object for the visualization.
    fontsize: float, default=14,
        x/y-axis label fontsize.
    cmap : str, default='rainbow'
        A registered colormap name used to map scalar data to colors.
    add_conjugate: bool, default=True,
        Plot the conjugate points on the uv plane.
    xlim, ylim: (xmin/ymin, xmax/ymax), default=(-9.5, 9.5)
        x-axis range in [Giga lambda] units
    shift_inital_time: bool,
        If True, observation time starts at t=0.0
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    giga = 10**9
    u = _np.concatenate([obsdata['u'] for obsdata in obs.tlist()]) / giga
    v = _np.concatenate([obsdata['v'] for obsdata in obs.tlist()]) / giga
    t = _np.concatenate([obsdata['time'] for obsdata in obs.tlist()])

    if shift_inital_time:
        t -= t.min()

    if add_conjugate:
        u = _np.concatenate([u, -u])
        v = _np.concatenate([v, -v])
        t = _np.concatenate([t, t])

    if ax is None:
        fig, ax = _plt.subplots(1, 1)
    else:
        fig = ax.get_figure()

    sc = ax.scatter(u, v, c=t, cmap=_plt.cm.get_cmap(cmap))
    ax.set_xlabel(r'East-West Freq $[G \lambda]$', fontsize=fontsize)
    ax.set_ylabel(r'North-South Freq $[G \lambda]$', fontsize=fontsize)
    ax.invert_xaxis()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3.5%', pad=0.2)
    cbar = fig.colorbar(sc, cax=cax, ticks=[0, 4, 8, 12])
    cbar.set_ticklabels(['{} Hrs'.format(tick) for tick in cbar.get_ticks()])
    _plt.tight_layout()

def obs_to_xarray(obs):
    """
    Generate an xr.Dataset from an ehtim.Observation object.

    Parameters
    ----------
    obs: ehtim.Observation
        An Observation object with 'vis' (visibilities) and 'sigma' (uncertainties).

    Returns
    -------
    visibilities: xr.Dataset
        Dataset of 'vis' and 'sigma'.

    """
    obslist = obs.tlist()
    u = _np.concatenate([obsdata['u'] for obsdata in obslist])
    v = _np.concatenate([obsdata['v'] for obsdata in obslist])
    t = _np.concatenate([obsdata['time'] for obsdata in obslist])
    visibilities = _xr.Dataset({'vis': ('index', obs.data['vis']), 'sigma': ('index', obs.data['sigma'])},
                               coords={'t': ('index', t), 'u': ('index', v), 'v': ('index', u),
                                       'uvdist': ('index', _np.sqrt(u ** 2 + v ** 2))})
    return visibilities

def observe_same(movie, obs, ttype='nfft', output_path='./caltable', thermal_noise=True, station_noise=False,
                 dterm_noise=False, sigmat=0.25, seed=False):
    """
    Generate an Obeservation object from the movie and add noise.

    Parameters
    ----------
    movie: ehtim.Movie or xr.DataArray
        Input movie. If the movie is an xr.DataArray object it is transformed into an ehtim.Movie first.
    obs: ehtim.Observation
        An (empty) Observation object
    output_path: str
        Output path for caltable
    thermal_noise: bool
        False for no thermal noise noise
    station_noise: bool,
        True for station based gain and phase errors
    dterm_noise: bool,
        True for dterm noise
    sigmat: float,
        Correlation time for random station based errors
    seed: int, default=6
        Seed for the random number generators, uses system time if False

    Returns
    -------
    obs: ehtim.Observation
        Observation object with visibilties of the input movie.
    """
    if isinstance(movie, _xr.DataArray):
        movie = movie.utils_observe.to_ehtim(source='SYNTHETIC')
    elif isinstance(movie, _eh.movie.Movie):
        pass
    else:
        raise AttributeError('Movie datatype ({}) not recognized.'.format(movie.__class__))

    # these gains are approximated from the EHT 2017 data
    # the standard deviation of the absolute gain of each telescope from a gain of 1
    GAIN_OFFSET = {'ALMA': 0.15, 'APEX': 0.15, 'SMT': 0.15, 'LMT': 0.6, 'PV': 0.15, 'SMA': 0.15, 'JCMT': 0.15,
                   'SPT': 0.15, 'SR': 0.0}
    GAINP = {'ALMA': 0.05, 'APEX': 0.05, 'SMT': 0.05, 'LMT': 0.5, 'PV': 0.05, 'SMA': 0.05, 'JCMT': 0.05,
             'SPT': 0.15, 'SR': 0.0}

    stabilize_scan_phase = True  # If true then add a single phase error for each scan to act similar to adhoc phasing
    stabilize_scan_amp = True    # If true then add a single gain error at each scan
    jones = True                 # Jones matrix corruption & calibration
    inv_jones = True             # Invert the jones matrix
    frcal = True                 # True if you do not include effects of field rotation
    dcal = not dterm_noise       # True if you do not include the effects of leakage
    if dterm_noise:
        dterm_offset = 0.05      # Random offset of D terms is given at each site with this std away from 1
    else:
        dterm_offset = _ehc.DTERMPDEF
    neggains = False
    if station_noise:
        ampcal = False          # If False, time-dependent gaussian errors are added to station gains
        phasecal = False        # If False, time-dependent station-based random phases are added
        rlgaincal = False       # If False, time-dependent gains are not equal for R and L pol
        gain_offset = GAIN_OFFSET
        gainp = GAINP
    else:
        ampcal = True
        phasecal = True
        rlgaincal = True
        gain_offset = _ehc.GAINPDEF
        gainp = _ehc.GAINPDEF
    movie.rf = obs.rf
    obs = movie.observe_same(obs, ttype=ttype, add_th_noise=thermal_noise, ampcal=ampcal, phasecal=phasecal,
                             stabilize_scan_phase=stabilize_scan_phase, stabilize_scan_amp=stabilize_scan_amp,
                             gain_offset=gain_offset, gainp=gainp, jones=jones, inv_jones=inv_jones,
                             dcal=dcal, frcal=frcal, rlgaincal=rlgaincal, neggains=neggains,
                             dterm_offset=dterm_offset, caltable_path=output_path, seed=seed, sigmat=sigmat)

    return obs

def empty_eht_obs(array, nt, tint, tstart=4.0, tstop=15.5,
                  ra=17.761121055553343, dec=-29.00784305556, rf=226191789062.5, mjd=57850,
                  bw=1856000000.0, timetype='UTC', polrep='stokes'):
    """
    Generate an empty ehtim.Observation from an array configuration and time constraints

    Parameters
    ----------
    array: ehtim.Array
        ehtim ehtim Array object (e.g. from: ehtim.array.load_txt(array_path))
    nt: int,
        Number of temporal frames.
    tint: float,
        Scan integration time in seconds
    tstart: float, default=4.0
        Start time of the observation in hours
    tstop: float, default=15.5
        End time of the observation in hours
    ra: float, default=17.761121055553343,
        Source Right Ascension in fractional hours.
    dec: float, default=-29.00784305556,
        Source declination in fractional degrees.
    rf: float, default=226191789062.5,
        Reference frequency observing at corresponding to 1.3 mm wavelength
    mjd: int, default=57850,
        Modified julian date of observation
    bw: float, default=1856000000.0,
    timetype: string, default='UTC',
        How to interpret tstart and tstop; either 'GMST' or 'UTC'
    polrep: sting, default='stokes',
        Polarization representation, either 'stokes' or 'circ'

    Returns
    -------
    obs: ehtim.Obsdata
        ehtim Observation object
    """
    tadv = (tstop - tstart) * 3600.0/ nt
    obs = array.obsdata(ra=ra, dec=dec, rf=rf, bw=bw, tint=tint, tadv=tadv, tstart=tstart, tstop=tstop, mjd=mjd,
                        timetype=timetype, polrep=polrep)
    return obs

def ehtim_to_xarray(ehtim_obj):
    """
    Transform ehtim movie or image to xarray.DataArray.

    Parameters
    ----------
    ehtim_obj: ehtim.image.Image or ehtim.movie.Movie object
        An Image or Movie object depending on the data dimensionality.

    Returns
    -------
    output: xr.DataArray
        A data array with the image or movie data and coordinates.

    Raises
    ------
    Attribute error if ehtim_obj is not supported.
    """
    grid = _utils.linspace_2d((ehtim_obj.ydim, ehtim_obj.xdim)).utils_image.set_fov(
        (ehtim_obj.fovx(), 'rad'))
    if isinstance(ehtim_obj, _eh.image.Image):
        data = ehtim_obj.imarr()
    elif isinstance(ehtim_obj, _eh.movie.Movie):
        data = ehtim_obj.iframes.reshape(ehtim_obj.nframes, ehtim_obj.xdim, ehtim_obj.ydim)
        grid.coords.update({'t': _xr.DataArray(ehtim_obj.times, dims='t', attrs={'units': 'UTC'})})
    else:
        raise AttributeError('Unsupported ehtim_obj: {}'.format(ehtim_obj.__class__))

    output = _xr.DataArray(data, dims=grid.dims, coords=grid.coords,
                           attrs={'ra': ehtim_obj.ra, 'dec': ehtim_obj.dec, 'rf': ehtim_obj.rf, 'mjd': ehtim_obj.mjd})
    return output

@_xr.register_dataarray_accessor("utils_observe")
class _ObserveAccessor(object):
    """
    Register a custom accessor ObserveAccessor on xarray.DataArray object.
    This adds methods for interfacing with eht-imaging classes and methods and computing Fourier domain quantities.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def eht_fillna(self, obs, fft_pad_factor=1, image_dims=['y', 'x'], fft_dims=['u', 'v'], add_conjugate=True):
        """Nearest neighbor sampling of EHT array measurements.
        The rest of the uv points are filled with np.nan values.

        Parameters
        ----------
        obs: Obsdata,
            ehtim observation data containing the array geometry and measurement times.
        fft_pad_factor: float, default=1
            Padding factor for increased fft resolution.
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.
        fft_dims: [dim1, dim2], default=['u', 'v']
            Fourier data dimensions.
        add_conjugate: bool, default=True,
            Add conjugate points on the uv plane.

        Returns
        -------
        output: xr.DataArray,
            A movie sampled with nearest neighbor interpolation on the uv point specified by obs.
            The rest of the uv-plane is filled with np.nan values.
        """
        fourier = self._obj.utils_observe.block_fourier(fft_pad_factor=fft_pad_factor, image_dims=image_dims,
                                                        fft_dims=fft_dims)

        u = _np.concatenate([obsdata['u'] for obsdata in obs.tlist()])
        v = _np.concatenate([obsdata['v'] for obsdata in obs.tlist()])
        t = _np.concatenate([obsdata['time'] for obsdata in obs.tlist()])

        # For static images duplicate temporal dimension
        if 't' not in fourier.coords:
            fourier = fourier.expand_dims(t=t)

        if add_conjugate:
            u = _np.concatenate([u, -u])
            v = _np.concatenate([v, -v])
            t = _np.concatenate([t, t])

        data = fourier.sel(t=_xr.DataArray(t, dims='index'), u=_xr.DataArray(v, dims='index'),
                           v=_xr.DataArray(u, dims='index'), method='nearest')

        output = _xr.full_like(fourier, fill_value=_np.nan)
        output.attrs.update(fourier.attrs)
        output.attrs.update(source=obs.source)
        output.loc[{'t': data.t, 'u': data.u, 'v': data.v}] = data
        return output

    def block_fourier(self, fft_pad_factor=2, image_dims=['y', 'x'], fft_dims=['u', 'v']):
        """Fast Fourier transform of one or several movies.
        Fourier is done per each time slice on image dimensions.
        This function adds a 2D triangle pulse and phase to match centroid convention.

        Parameters
        ----------
        fft_pad_factor: float, default=2
            Padding factor for increased fft resolution.
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.
        fft_dims: [dim1, dim2], default=['u', 'v']
            Fourier data dimensions.

        Returns
        -------
        block_fft: xr.DataArray,
            A data array of shape (num_movies, num_visibilities) and dtype np.complex128.

        Notes
        -----
        Units of input data image_dims must be 'rad', 'as' or uas'.
        For high order downstream interpolation (e.g. cubic) of uv points a higher fft_pad_factor should be taken.
        """
        movies = self._obj.squeeze().utils_image.change_units(image_dims=image_dims)

        psize = movies.utils_image.psize
        image_dim_sizes = (movies[image_dims[0]].size, movies[image_dims[1]].size)
        fft = movies.utils_fourier.fft(fft_pad_factor, image_dims, fft_dims)
        fft *= fft.utils_fourier._trianglePulse2D(fft.u, fft.v, psize) * \
               fft.utils_fourier._extra_phase(fft.u, fft.v, psize, image_dim_sizes)

        if ('r' in fft.coords) and ('theta' in fft.coords):
            fft = fft.utils_polar.add_coords(image_dims=['u', 'v'])
        return fft

    def block_observe_same_nonoise(self, obs, fft_pad_factor=2, image_dims=['y', 'x'], max_mbs=1000.0):
        """
        Modification of ehtim's observe_same_nonoise method for a block of movies.

        Parameters
        ----------
        obs: Obsdata,
            ehtim observation data containing the array geometry and measurement times.
        fft_pad_factor: float default=2
            Padding factor for increased fft resolution.
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.
        max_mbs: float, default=300.0
            Maximum number of Megabytes to allocate for fft. If fft exceeds this an attempt to loop over non-spatial
            direction is made

        Returns
        -------
        visibilities: xr.DataArray
            DataArray of visibilities, shape = (num_movies, num_visibilities) and dtype np.complex128.

        Notes
        -----
        1. NFFT is not supported.
        2. Cubic interpolation in both time and uv (ehtim has linear interpolation in time and cubic in uv).
        3. Units of input data image_dims must be 'rad', 'as' or uas'.
        4. Due to cubic interpolation of nd.map_coordinates fft_pad_factor should be >= 2.

        References
        ----------
        https://github.com/achael/eht-imaging/blob/6b87cdc65bdefa4d9c4530ea6b69df9adc531c0c/ehtim/movie.py#L981
        https://github.com/achael/eht-imaging/blob/6b87cdc65bdefa4d9c4530ea6b69df9adc531c0c/ehtim/observing/obs_simulate.py#L182
        """
        movies = self._obj.utils_image.change_units(image_dims=image_dims)
        movies.utils_movie.check_time_units(obs.timetype)

        fft_dims = ['u', 'v']
        psize = movies.utils_image.psize
        image_dim_sizes = (movies[image_dims[0]].size, movies[image_dims[1]].size)


        # Check if fft fits in memory or split into chunks
        mbs = 1e-6
        fft_size = _np.prod(movies.shape) * _utils.next_power_of_two(fft_pad_factor) ** 2
        fft_size_mbs = fft_size * _np.dtype(_np.complex).itemsize * mbs

        # Split into chunks along largest dimension which is not a movie dimension
        other_dims = list(set(movies.sizes) - set(image_dims) - set(['t']))

        num_chunks = 1
        if (fft_size_mbs > max_mbs):
            if (not other_dims):
                raise AttributeError('Not enough space for fft: fft_size_mbs={}, max_mbs={}'.format(fft_size_mbs, max_mbs))
            else:
                split_dim = other_dims[_np.argmax([movies.sizes[dim] for dim in other_dims])]
                split_dim_size = movies.sizes[split_dim]
                chunk_size = min(int(split_dim_size / (fft_size_mbs / max_mbs)), split_dim_size)
                num_chunks = int(_np.ceil(split_dim_size / chunk_size))

        output = []
        for i in range(num_chunks):
            if num_chunks == 1:
                fft = movies.utils_fourier.fft(fft_pad_factor, image_dims, fft_dims=fft_dims)
            elif num_chunks > 1:
                fft = movies.isel({split_dim: slice(i * chunk_size, (i + 1) * chunk_size)}).utils_fourier.fft(
                    fft_pad_factor, image_dims, fft_dims=fft_dims)
            else:
                raise AttributeError('illegal num_chunks')

            obslist = obs.tlist()
            u = _np.concatenate([obsdata['u'] for obsdata in obslist])
            v = _np.concatenate([obsdata['v'] for obsdata in obslist])
            t = _np.concatenate([obsdata['time'] for obsdata in obslist])

            # Extra phase to match centroid convention
            pulsefac = fft.utils_fourier._trianglePulse2D(u, v, psize)
            phase = fft.utils_fourier._extra_phase(u, v, psize, image_dim_sizes)

            fft = fft.assign_coords(u2=(fft_dims[0], range(fft[fft_dims[0]].size)),
                                    v2=(fft_dims[1], range(fft[fft_dims[1]].size)))
            tuv2 = _np.vstack((fft.v2.interp({fft_dims[1]: v}), fft.u2.interp({fft_dims[0]: u})))

            if 't' in fft.coords:
                fft = fft.assign_coords(t2=('t', range(fft.t.size)))
                tuv2 = _np.vstack((fft.t2.interp(t=t), tuv2))

            if (fft.ndim == 2) or (fft.ndim == 3):
                visre = _nd.map_coordinates(_np.ascontiguousarray(_np.real(fft).data), tuv2)
                visim = _nd.map_coordinates(_np.ascontiguousarray(_np.imag(fft).data), tuv2)
                vis = visre + 1j * visim
                visibilities = _xr.DataArray(vis * phase * pulsefac, dims='index')

            elif fft.ndim > 3:
                # Sample block Fourier on tuv coordinates of the observations
                # Note: using np.ascontiguousarray preforms ~twice as fast
                # Concatenate first axis as a block
                fft_shape = fft.shape
                fft = fft.data.reshape(-1, *fft_shape[-3:])
                num_block = fft.shape[0]
                num_vis = len(t)
                visibilities = _np.empty(shape=(num_block, num_vis), dtype=_np.complex128)
                for k, fourier in enumerate(fft):
                    visre = _nd.map_coordinates(_np.ascontiguousarray(_np.real(fourier).data), tuv2)
                    visim = _nd.map_coordinates(_np.ascontiguousarray(_np.imag(fourier).data), tuv2)
                    vis = visre + 1j * visim
                    visibilities[k] = vis * phase * pulsefac

                dims = movies.dims[:-3] + ('index',)
                coords = dict(zip(movies.dims[:-3], [movies.coords[dim] for dim in movies.dims[:-3]]))
                if num_chunks > 1:
                    coords[split_dim] = movies.isel(
                        {split_dim: slice(i * chunk_size, (i + 1) * chunk_size)}).coords[split_dim]

                visibilities = _xr.DataArray(visibilities.reshape(*fft_shape[:-3], num_vis), dims=dims, coords=coords)

            else:
                raise ValueError("unsupported number of dimensions: {}".format(fft.ndim))

            visibilities = visibilities.assign_coords(
                t=('index', t), u=('index', v), v=('index', u), uvdist=('index', _np.sqrt(u**2 + v**2)))
            visibilities.attrs.update(fft_pad_factor=fft_pad_factor, source=obs.source)
            output.append(visibilities)

            del fft
            _gc.collect()

        visibilities = _xr.concat(output, dim=split_dim) if num_chunks > 1 else output[0]
        return visibilities

    def to_ehtim(self, ra=17.761121055553343, dec=-29.00784305556, rf=226191789062.5, mjd=57850, source='SGRA',
                 image_dims=['y', 'x']):
        """
        Transform xarray movie to eht-imaging Movie object. Defaults to SgrA*.

        Parameters
        ----------
        ra: float, default=17.761121055553343,
            Source Right Ascension in fractional hours.
        dec: float, default=-29.00784305556,
            Source declination in fractional degrees.
        rf: float, default=226191789062.5,
            Reference frequency observing at corresponding to 1.3 mm wavelength
        mjd: int, default=57850,
            Modified julian date of observation
        source: str, default='SGRA',
            Source name
        image_dims: [dim1, dim2], default=['y', 'x'],
            Image data dimensions.

        Returns
        -------
        output: ehtim.Movie or ehtim.Image objects
            A Movie or Image object depending on the data dimensionality
        """
        movie = self._obj.squeeze().utils_image.change_units(image_dims=image_dims)
        movie.utils_movie.check_time_units('UTC')

        if movie.ndim > 3:
            raise AttributeError('Data dimensions are greater than 3.')
        im_list = []
        if 't' in movie.dims:
            for i, time in enumerate(movie.t):
                frame = movie.isel(t=i)
                image = _eh.image.make_empty(frame.sizes[image_dims[0]], movie.utils_image.fov[0], ra, dec, rf, mjd=mjd,
                                             source=source)
                image.time = float(time)
                image.ivec = frame.data.ravel()
                im_list.append(image)
            output = _eh.movie.merge_im_list(im_list)
        else:
            output = _eh.image.make_empty(movie.sizes[image_dims[0]], movie.utils_image.fov[0], ra, dec, rf, mjd=mjd,
                                          source=source)
            output.ivec = movie.data.ravel()
        return output

