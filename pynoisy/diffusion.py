"""
TODO: Some documentation and general description goes here.
"""
import core
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage.transform
from scipy import interpolate
from pynoisy import RotationDirection, Image


class Diffusion(Image):
    """TODO"""
    def __init__(self, principle_angle=None, diffusion_coefficient=None, correlation_time=None, tensor_ratio=1.0):
        super().__init__()
        self._principle_angle = np.empty(shape=self.image_size, dtype=np.float64)  if principle_angle is None else np.array(principle_angle, dtype=np.float64, order='C')
        self._diffusion_coefficient = np.empty(shape=self.image_size, dtype=np.float64) if diffusion_coefficient is None else np.array(diffusion_coefficient, dtype=np.float64, order='C')
        self._correlation_time = np.empty(shape=self.image_size, dtype=np.float64) if correlation_time is None else np.array(correlation_time, dtype=np.float64, order='C')
        self._correlation_length = None
        self._tensor_ratio = tensor_ratio

    def set_correlation_length(self, correlation_length):
        """TODO"""
        self._correlation_length = correlation_length
        self._diffusion_coefficient = 2.0 * correlation_length**2 / self._correlation_time
        print('Updating diffusion coefficient according to correlation length and time: D = 2.0 * l**2 / t')

    def get_tensor(self):
        return core.get_diffusion_tensor(self.tensor_ratio, self.principle_angle, self.diffusion_coefficient)

    def get_laplacian(self, frames, advection_image):
        lap = core.get_laplacian(
            self.tensor_ratio,
            self.principle_angle,
            self.diffusion_coefficient,
            np.array(advection_image, dtype=np.float64, order='C'),
            self.correlation_time,
            np.array(frames, dtype=np.float64, order='C')
        )
        return lap

    def __add__(self, other):
        x = np.linspace(min(self.x.min(), other.x.min()), max(self.x.max(), other.x.max()), self.image_size[0])
        y = np.linspace(min(self.y.min(), other.y.min()), max(self.y.max(), other.y.max()), self.image_size[1])
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
        new_shape = (downscale_factor * np.array(self.image_size)).astype(int)
        vx = skimage.transform.resize(np.cos(self.principle_angle), new_shape)
        vy = skimage.transform.resize(np.sin(self.principle_angle), new_shape)
        scaled_x = skimage.transform.resize(self.x, new_shape)
        scaled_y = skimage.transform.resize(self.y, new_shape)
        plt.quiver(scaled_x, scaled_y, vx, vy)
        plt.title('Diffusion tensor (primary)', fontsize=18)

    def plot_secondary_axis(self, downscale_factor=0.125):
        """TODO"""
        new_shape = (downscale_factor * np.array(self.image_size)).astype(int)
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
        self._opening_angle = opening_angle
        self._tau = tau
        self._lam = lam
        self._scaling_radius = scaling_radius
        self._tensor_ratio = tensor_ratio
        self._principle_angle = core.get_disk_angle(opening_angle)
        self._correlation_time = core.get_disk_correlation_time(tau, scaling_radius)
        self._diffusion_coefficient = core.get_disk_diffusion_coefficient(tau, lam, scaling_radius)
        self._correlation_length =  core.get_disk_correlation_length(scaling_radius, lam)

    @property
    def opening_angle(self):
        return self._opening_angle

    @property
    def tau(self):
        return self._tau

    @property
    def lam(self):
        return self._lam

    @property
    def scaling_radius(self):
        return self._scaling_radius


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