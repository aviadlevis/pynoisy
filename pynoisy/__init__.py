"""
TODO: Some documentation and general description goes here.
"""
from enum import Enum
import core
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable


class RotationDirection(Enum):
    clockwise = -1.0
    counter_clockwise = 1.0


class Image(object):
    """TODO"""
    def __init__(self):
        self._x = np.empty(shape=core.get_image_size(), dtype=np.float64)
        self._y = np.empty(shape=core.get_image_size(), dtype=np.float64)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


class Advection(Image):
    """TODO"""
    def __init__(self):
        super().__init__()
        self._v = np.empty(shape=core.get_image_size() + [2], dtype=np.float64)

    def plot_velocity(self, downscale_factor=0.125):
        """TODO"""
        scaled_v = scipy.ndimage.zoom(self.v, (downscale_factor, downscale_factor, 1))
        scaled_x = scipy.ndimage.zoom(self.x, downscale_factor)
        scaled_y = scipy.ndimage.zoom(self.y, downscale_factor)
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
        self._x, self._y = core.get_xy_grid()


class Diffusion(Image):
    """TODO"""
    def __init__(self):
        super().__init__()
        self._principle_angle = np.empty(shape=core.get_image_size(), dtype=np.float64)
        self._diffusion_coefficient = np.empty(shape=core.get_image_size(), dtype=np.float64)
        self._correlation_time = np.empty(shape=core.get_image_size(), dtype=np.float64)
        self._tensor_ratio = 1.0

    def plot_principal_axis(self, downscale_factor=0.125):
        """TODO"""
        scaled_phi = scipy.ndimage.zoom(self.principle_angle, downscale_factor)
        scaled_x = scipy.ndimage.zoom(self.x, downscale_factor)
        scaled_y = scipy.ndimage.zoom(self.y, downscale_factor)
        vx = np.cos(scaled_phi)
        vy = np.sin(scaled_phi)
        plt.quiver(scaled_x, scaled_y, vx, vy)
        plt.title('Diffusion tensor (primary)', fontsize=18)
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.5, 0.5])

    def plot_secondary_axis(self, downscale_factor=0.125):
        """TODO"""
        scaled_phi = scipy.ndimage.zoom(self.principle_angle, downscale_factor)
        scaled_x = scipy.ndimage.zoom(self.x, downscale_factor)
        scaled_y = scipy.ndimage.zoom(self.y, downscale_factor)
        vx = -np.sin(scaled_phi)
        vy = np.cos(scaled_phi)
        plt.quiver(scaled_x, scaled_y, vx, vy)
        plt.title('Diffusion tensor (secondary)', fontsize=18)
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.5, 0.5])

    def imshow_correlation_time(self):
        """TODO"""
        im = plt.imshow(self.correlation_time)
        plt.title('Correlation time', fontsize=18)
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


class DiskDiffusion(Diffusion):
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
        self._tensor_ratio = tensor_ratio
        self._x, self._y = core.get_xy_grid()


class Envelope(Image):
    """
    The envelope function multiplies the random fields to create synthetic images.
    The equation for each field is: image = exp(-amplitude*del)*envelope

    Parameters
    ----------
    amplitude: float, default = 0.05
        strength of perturbation; image = exp(-amplitude*del)*envelope
    """
    def __init__(self, amplitude=0.05):
        super().__init__()
        self._envelope = np.ones(shape=core.get_image_size(), dtype=np.float64, order='C')
        self._amplitude = amplitude

    def imshow(self):
        """TODO"""
        im = plt.imshow(self.envelope)
        plt.title('Envelope function', fontsize=18)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    @property
    def envelope(self):
        return self._envelope

    @property
    def amplitude(self):
        return self._amplitude


class DiskEnvelope(Envelope):
    """
    The envelope function multiplies the random fields to create synthetic images.
    The equation for each field is: image = exp(-amplitude*del)*envelope
    The disk envelope function is a specific envelope defined by the src/model_disk.c

    Parameters
    ----------
    amplitude: float, default = 0.05
        strength of perturbation; image = exp(-amplitude*del)*envelope
    """
    def __init__(self, amplitude=0.05, scaling_radius=0.2):
        super().__init__(amplitude)
        self._envelope = core.get_disk_envelope(scaling_radius)
        self._x, self._y = core.get_xy_grid()


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

    def run(self, evolution_length=0.1):
        """TODO"""
        core.run_main(
            self.diffusion.tensor_ratio,
            self.envelope.amplitude,
            self.forcing_strength,
            evolution_length,
            self.diffusion.principle_angle,
            self.advection.v,
            self.diffusion.diffusion_coefficient,
            self.diffusion.correlation_time,
            self.envelope.envelope
        )

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