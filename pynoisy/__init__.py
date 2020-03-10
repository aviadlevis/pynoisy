"""
TODO: Some documentation and general description goes here.
"""
from enum import Enum
import core
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import os
import skimage.transform
from scipy import interpolate
import copy

class RotationDirection(Enum):
    clockwise = -1.0
    counter_clockwise = 1.0


class Image(object):
    """TODO"""
    def __init__(self):
        self._x, self._y = core.get_xy_grid()
        self._image_shape = core.get_image_size()
        self._r = np.sqrt(self.x ** 2 + self.y ** 2)

    def shift(self, x0, y0):
        """TODO"""
        self._x += x0
        self._y += y0

    @property
    def image_shape(self):
        return self._image_shape

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def r(self):
        return self._r


class Advection(Image):
    """TODO"""
    def __init__(self, velocity=None):
        super().__init__()
        self._v = np.empty(shape=self.image_shape + [2], dtype=np.float64) if velocity is None else np.array(velocity, dtype=np.float64, order='C')

    def plot_velocity(self, downscale_factor=0.125):
        """TODO"""
        new_shape = (downscale_factor * np.array(self.image_shape)).astype(int)
        scaled_v = skimage.transform.resize(self.v, new_shape)
        scaled_x = skimage.transform.resize(self.x, new_shape)
        scaled_y = skimage.transform.resize(self.y, new_shape)
        plt.quiver(scaled_x, scaled_y, scaled_v[..., 0], scaled_v[..., 1])
        plt.title('Disk velocity field (Kepler induced)', fontsize=18)
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.5, 0.5])

    @property
    def v(self):
        return self._v

    @property
    def vx(self):
        return self._v[...,0]

    @property
    def vy(self):
        return self._v[...,0]


class DiskAdvection(Advection):
    """
    TODO

    Parameters
    ----------
    direction: RotationDirection, , default=clockwise
        clockwise or counter clockwise
    scaling_radius: float, default=0.2
        scaling parameter for the Kepler orbital frequency (the magnitude of the velocity)
    """
    def __init__(self, direction=RotationDirection.clockwise, scaling_radius=0.2):
        super().__init__()
        self._v = core.get_disk_velocity(direction.value, scaling_radius)

    def __add__(self, other):
        """TODO"""
        x = np.linspace(min(self.x.min(), other.x.min()), max(self.x.max(), other.x.max()), self.image_shape[0])
        y = np.linspace(min(self.y.min(), other.y.min()), max(self.y.max(), other.y.max()), self.image_shape[1])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        v1x = interpolate.interp2d(self.x[:, 0], self.y[0], self.v[...,0], bounds_error=False, fill_value=0.0)
        v2x = interpolate.interp2d(other.x[:, 0], other.y[0], other.v[...,0], bounds_error=False, fill_value=0.0)
        v1y = interpolate.interp2d(self.x[:, 0], self.y[0], self.v[...,1], bounds_error=False, fill_value=0.0)
        v2y = interpolate.interp2d(other.x[:, 0], other.y[0], other.v[...,1], bounds_error=False, fill_value=0.0)
        velocity = np.stack(((v1x(x, y) + v2x(x, y)) / 2.0, (v1y(x, y) + v2y(x, y)) / 2.0), axis=-1)
        advection = Advection(velocity)
        advection._x = np.array(xx, dtype=np.float64, order='C')
        advection._y = np.array(yy, dtype=np.float64, order='C')
        return advection


class Diffusion(Image):
    """TODO"""
    def __init__(self, principle_angle=None, diffusion_coefficient=None, correlation_time=None, tensor_ratio=1.0):
        super().__init__()
        self._principle_angle = np.empty(shape=self.image_shape, dtype=np.float64)  if principle_angle is None else np.array(principle_angle, dtype=np.float64, order='C')
        self._diffusion_coefficient = np.empty(shape=self.image_shape, dtype=np.float64) if diffusion_coefficient is None else np.array(diffusion_coefficient, dtype=np.float64, order='C')
        self._correlation_time = np.empty(shape=self.image_shape, dtype=np.float64) if correlation_time is None else np.array(correlation_time, dtype=np.float64, order='C')
        self._correlation_length = None
        self._tensor_ratio = tensor_ratio

    def set_correlation_length(self, correlation_length):
        """TODO"""
        self._correlation_length = correlation_length
        self._diffusion_coefficient = 2.0 * correlation_length**2 / self._correlation_time
        print('Updating diffusion coefficient according to correlation length and time: D = 2.0 * l**2 / t')

    def __add__(self, other):
        x = np.linspace(min(self.x.min(), other.x.min()), max(self.x.max(), other.x.max()), self.image_shape[0])
        y = np.linspace(min(self.y.min(), other.y.min()), max(self.y.max(), other.y.max()), self.image_shape[1])
        xx, yy = np.meshgrid(x, y, indexing='ij')

        dc1 = interpolate.interp2d(self.x[:,0], self.y[0], self.diffusion_coefficient, bounds_error=False, fill_value = 0.0)
        dc2 = interpolate.interp2d(other.x[:,0], other.y[0], other.diffusion_coefficient, bounds_error=False, fill_value = 0.0)
        ct1 = interpolate.interp2d(self.x[:, 0], self.y[0], self.correlation_time, bounds_error=False,  fill_value=0.0)
        ct2 = interpolate.interp2d(other.x[:, 0], other.y[0], other.correlation_time, bounds_error=False, fill_value=0.0)
        cos1 = interpolate.interp2d(self.x[:, 0], self.y[0], np.cos(self.principle_angle), bounds_error=False, fill_value=0.0)
        cos2 = interpolate.interp2d(other.x[:, 0], other.y[0], np.cos(other.principle_angle), bounds_error=False, fill_value=0.0)
        sin1 = interpolate.interp2d(self.x[:, 0], self.y[0], np.sin(self.principle_angle), bounds_error=False, fill_value=0.0)
        sin2 = interpolate.interp2d(other.x[:, 0], other.y[0], np.sin(other.principle_angle), bounds_error=False, fill_value=0.0)

        diffusion_coefficient = dc1(x, y) + dc2(x, y)
        cos = (dc1(x, y) * cos1(x, y) + dc2(x, y) * cos2(x, y)) / (diffusion_coefficient + 1e-8)
        sin = (dc1(x, y) * sin1(x, y) + dc2(x, y) * sin2(x, y)) / (diffusion_coefficient + 1e-8)
        principle_angle = np.arctan2(sin, cos)
        correlation_time = ct1(x, y) + ct2(x, y)
        diffusion = Diffusion(principle_angle, diffusion_coefficient, correlation_time, self.tensor_ratio)
        diffusion._x = np.array(xx, dtype=np.float64, order='C')
        diffusion._y = np.array(yy, dtype=np.float64, order='C')

        return diffusion

    def __setitem__(self, indices, data):
        self._principle_angle[indices] = data.principle_angle[indices]
        self._diffusion_coefficient[indices] = data.diffusion_coefficient[indices]
        self._correlation_time[indices] = data.correlation_time[indices]
        if self.correlation_length is not None and data.correlation_length is not None:
            self._correlation_length[indices] = data.correlation_length[indices]

    def plot_principal_axis(self, downscale_factor=0.125):
        """TODO"""
        new_shape = (downscale_factor * np.array(self.image_shape)).astype(int)
        vx = skimage.transform.resize(np.cos(self.principle_angle), new_shape)
        vy = skimage.transform.resize(np.sin(self.principle_angle), new_shape)
        scaled_x = skimage.transform.resize(self.x, new_shape)
        scaled_y = skimage.transform.resize(self.y, new_shape)
        plt.quiver(scaled_x, scaled_y, vx, vy)
        plt.title('Diffusion tensor (primary)', fontsize=18)

    def plot_secondary_axis(self, downscale_factor=0.125):
        """TODO"""
        new_shape = (downscale_factor * np.array(self.image_shape)).astype(int)
        vx = skimage.transform.resize(-np.sin(self.principle_angle), new_shape)
        vy = skimage.transform.resize(np.cos(self.principle_angle), new_shape)
        scaled_x = skimage.transform.resize(self.x, new_shape)
        scaled_y = skimage.transform.resize(self.y, new_shape)
        plt.quiver(scaled_x, scaled_y, vx, vy)
        plt.title('Diffusion tensor (secondary)', fontsize=18)

    def imshow_correlation_time(self):
        """TODO"""
        im = plt.imshow(self.correlation_time)
        plt.title('Correlation time', fontsize=18)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    def imshow_correlation_length(self):
        """TODO"""
        if self.correlation_length is None:
            print('Correlation length is not set')
        else:
            im = plt.imshow(self.correlation_length)
            plt.title('Correlation length', fontsize=18)
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax)

    def imshow_diffusion_coefficient(self):
        """TODO"""
        im = plt.imshow(self.diffusion_coefficient)
        plt.title('Diffusion coefficient (primary axis)', fontsize=18)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    @property
    def tensor_ratio(self):
        return self._tensor_ratio

    @property
    def principle_angle(self):
        return self._principle_angle

    @property
    def diffusion_coefficient(self):
        return self._diffusion_coefficient

    @property
    def correlation_time(self):
        return self._correlation_time

    @property
    def correlation_length(self):
        return self._correlation_length


class RingDiffusion(Diffusion):
    """
    TODO

    Parameters
    ----------
    opening_angle: float, default= pi/3
        This angle defines the opening angle of spirals with respect to the local radius
    tau: float, default=1.0
        product of correlation time and local Keplerian frequency
    lam: float, default=0.5
        ratio of correlation length to local radius
    scaling_radius: float, default=0.2
        scaling parameter for the Kepler orbital frequency (the magnitude of the velocity)
    tensor_ratio: float, default=0.1
        ratio for the diffusion tensor along the two principal axis.
    """
    def __init__(self,
                 opening_angle=np.pi/3.0,
                 tau=1.0,
                 lam=0.5,
                 scaling_radius=0.2,
                 tensor_ratio=0.1):
        super().__init__()
        self._principle_angle = core.get_disk_angle(opening_angle)
        self._correlation_time = core.get_disk_correlation_time(tau, scaling_radius)
        self._diffusion_coefficient = core.get_disk_diffusion_coefficient(tau, lam, scaling_radius)
        self._correlation_length =  core.get_disk_correlation_length(scaling_radius, lam)
        self._tensor_ratio = tensor_ratio


class DiskDiffusion(Diffusion):
    """
    TODO

    Parameters
    ----------
    tau: float, default=1.0
        product of correlation time and local Keplerian frequency
    scaling_radius: float, default=0.2
        scaling parameter for the Kepler orbital frequency (the magnitude of the velocity)
    tensor_ratio: float, default=0.1
        ratio for the diffusion tensor along the two principal axis.
    """
    def __init__(self, direction=RotationDirection.clockwise, tau=1.0, scaling_radius=0.2, tensor_ratio=0.1):
        super().__init__()
        self._principle_angle = core.get_disk_angle(-direction.value * np.pi/2)
        self._correlation_time = core.get_disk_correlation_time(tau, scaling_radius)
        self._diffusion_coefficient = np.exp(-0.5*(self.r/scaling_radius)**2)
        self._tensor_ratio = tensor_ratio


class Envelope(Image):
    """
    The envelope function multiplies the random fields to create synthetic images.
    The equation for each field is: image = exp(-amplitude*del)*envelope

    Parameters
    ----------
    amplitude: float, default = 0.05
        strength of perturbation; image = exp(-amplitude*del)*envelope
    """
    def __init__(self, data=None, amplitude=0.05):
        super().__init__()
        self._data = np.ones(shape=self.image_shape, dtype=np.float64, order='C') if data is None else data
        self._amplitude = amplitude

    def __add__(self, other):
        x = np.linspace(min(self.x.min(), other.x.min()), max(self.x.max(), other.x.max()), self.image_shape[0])
        y = np.linspace(min(self.y.min(), other.y.min()), max(self.y.max(), other.y.max()), self.image_shape[1])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        f1 = interpolate.interp2d(self.x[:,0], self.y[0], self.data, bounds_error=False, fill_value = 0.0)
        f2 = interpolate.interp2d(other.x[:,0], other.y[0], other.data, bounds_error=False, fill_value = 0.0)
        envelope = Envelope(amplitude=self.amplitude)
        envelope._x = xx
        envelope._y = yy
        envelope._data = np.array(f1(x, y) + f2(x, y), dtype=np.float64, order='C')
        return envelope

    def __mul__(self, other):
        envelope = copy.copy(self)
        envelope._data = np.array(self.data * other, dtype=np.float64, order='C')
        return envelope

    def __truediv__(self, other):
        envelope = copy.copy(self)
        envelope._data /= other
        return envelope

    def imshow(self):
        """TODO"""
        im = plt.imshow(self.data)
        plt.title('Envelope function', fontsize=18)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    @property
    def data(self):
        return self._data

    @property
    def amplitude(self):
        return self._amplitude


class RingEnvelope(Envelope):
    """
    The envelope function multiplies the random fields to create synthetic images.
    The equation for each field is: image = exp(-amplitude*del)*envelope

    Parameters
    ----------
    amplitude: float, default = 0.05
        strength of perturbation; image = exp(-amplitude*del)*envelope
    """
    def __init__(self, inner_radius=0.2, outer_radius=1.0, photon_ring_thickness=0.05, photon_ring_contrast=0.95,
                 photon_ring_decay=100.0, ascent=1.0, inner_decay=5.0, outer_decay=10, amplitude=0.05):
        super().__init__(amplitude)

        zone0_radius = inner_radius
        zone1_radius = inner_radius + photon_ring_thickness

        decay1 = photon_ring_decay
        decay2 = inner_decay
        decay3 = outer_decay

        zone0 = np.exp(-1.0 / ((self.r + 1e-8) / (ascent * zone0_radius * 2)) ** 2)
        zone0[self.r > zone0_radius] = 0

        zone1 = (photon_ring_contrast + np.exp(-decay1 * (self.r - zone0_radius))) * np.exp(-1.0 / ((zone0_radius + 1e-8) / (ascent * zone0_radius * 2)) ** 2)
        zone1[self.r <= zone0_radius] = 0
        zone1[self.r > zone1_radius] = 0

        zone2 = np.exp(-decay2 * (self.r - zone1_radius)) * np.exp(-1.0 / ((zone0_radius + 1e-8) / (ascent * zone0_radius * 2)) ** 2)  * \
                (photon_ring_contrast + np.exp(-decay1 * (zone1_radius - zone0_radius)))
        zone2[self.r <= zone1_radius] = 0

        data = zone0 + zone1 + zone2

        if outer_radius < 1.0:
            data[self.r > outer_radius] = 0
            zone3 = np.exp(-decay3 * (self.r - outer_radius)) * np.exp(-decay2 * (outer_radius - zone1_radius)) * \
                    np.exp(-1.0 / ((zone0_radius + 1e-8) / (ascent * zone0_radius * 2)) ** 2)  * \
                    (photon_ring_contrast + np.exp(-decay1 * (zone1_radius - zone0_radius)))
            zone3[self.r <= outer_radius] = 0
            data += zone3

        self._data = np.array(data, dtype=np.float64, order='C')


class NoisyDiskEnvelope(Envelope):
    """
    The envelope function multiplies the random fields to create synthetic images.
    The equation for each field is: image = exp(-amplitude*del)*envelope
    The disk envelope function is a specific envelope defined by the src/model_disk.c

    Parameters
    ----------
    amplitude: float, default = 0.05
        strength of perturbation; image = exp(-amplitude*del)*envelope
    scaling_radius: float, default=0.02
        Scales the disk radius with respect to the whole image
    """
    def __init__(self, amplitude=0.05, scaling_radius=0.2):
        super().__init__(amplitude)
        self._data = core.get_disk_envelope(scaling_radius)


class GaussianEnvelope(Envelope):
    """
    The envelope function multiplies the random fields to create synthetic images.
    The equation for each field is: image = exp(-amplitude*del)*envelope

    Parameters
    ----------
    amplitude: float, default = 0.05
        strength of perturbation; image = exp(-amplitude*del)*envelope
    """
    def __init__(self, std=0.2, FWHM=None, amplitude=0.05):
        super().__init__(amplitude)
        std2FWHM = lambda std: std * np.sqrt(2 * np.log(2)) * 2 / np.sqrt(2)
        FWHM2std = lambda gamma: gamma * np.sqrt(2) / (np.sqrt(2 * np.log(2)) * 2)

        if FWHM is None:
            self._std, self._FWHM = std, std2FWHM(std)
        else:
            self._std, self._FWHM = FWHM2std(FWHM), FWHM
        data = np.exp(-(self.r / self.std) ** 2)
        self._data = np.array(data, dtype=np.float64, order='C')

    @property
    def std(self):
        return self._std

    @property
    def FWHM(self):
        return self._FWHM


class DiskEnvelope(Envelope):
    """
    The envelope function multiplies the random fields to create synthetic images.
    The equation for each field is: image = exp(-amplitude*del)*envelope

    Parameters
    ----------
    amplitude: float, default = 0.05
        strength of perturbation; image = exp(-amplitude*del)*envelope
    """
    def __init__(self, radius=0.2, decay=20, amplitude=0.05):
        super().__init__(amplitude)
        data = np.ones_like(self.r, dtype=np.float64, order='C')
        data[self.r >= .95 * radius] = 0
        exponential_decay = np.exp(-decay * (self.r - .95 * radius))
        data[self.r >= .95 * radius] = exponential_decay[self.r >= .95 * radius]
        self._data = np.array(data, dtype=np.float64, order='C')


    @property
    def std(self):
        return self._std

    @property
    def FWHM(self):
        return self._FWHM


class PDESolver(object):
    """TODO"""
    def __init__(self, advection=Advection(), diffusion=Diffusion(), envelope=Envelope(), forcing_strength=1.0):
        self._forcing_strength = forcing_strength
        self.set_advection(advection)
        self.set_diffusion(diffusion)
        self.set_envelope(envelope)

    def set_advection(self, advection):
        """TODO"""
        self._advection = advection

    def set_diffusion(self, diffusion):
        """TODO"""
        self._diffusion = diffusion

    def set_envelope(self, envelope):
        """TODO"""
        self._envelope = envelope

    def run(self, evolution_length=0.1, verbose=True):
        """TODO"""
        frames = core.run_main(
            self.diffusion.tensor_ratio,
            self.envelope.amplitude,
            self.forcing_strength,
            evolution_length,
            self.diffusion.principle_angle,
            self.advection.v,
            self.diffusion.diffusion_coefficient,
            self.diffusion.correlation_time,
            self.envelope.data,
            verbose
        )
        return Movie(frames, duration=evolution_length)

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
    def envelope(self):
        return self._envelope


class Movie(object):
    """TODO"""

    def __init__(self, frames=None, duration=None):
        self._frames = frames
        self._duration = duration

    def duplicate_single_frame(self, frame, num_frames):
        self._frames = [frame] * num_frames

    def save(self, path):
        """Save movie to file.

        Args:
            path (str): Full path to file.
        """

        # safe creation of the directory
        directory = path[:-(1+path[::-1].find('/'))]
        if not os.path.exists(directory):
            os.makedirs(directory)

        file = open(path, 'wb')
        file.write(pickle.dumps(self.__dict__, -1))
        file.close()

    def load(self, path):
        """Load movie from file.

        Args:
            path (str): Full path to file.
        """
        file = open(path, 'rb')
        data = file.read()
        file.close()
        self.__dict__ = pickle.loads(data)

    @property
    def duration(self):
        return self._duration

    @property
    def frames(self):
        return self._frames

    @property
    def npix(self):
        return self.frames.shape[1]

    @property
    def num_frames(self):
        return len(self._frames)

    @property
    def frame_duration(self):
        if self.duration is not None and self.num_frames is not None:
            return self.duration / self.num_frames
        else:
            return None