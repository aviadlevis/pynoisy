"""
TODO: Some documentation and general description goes here.
"""
import core
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from scipy import interpolate
from pynoisy import Image, RotationDirection


class Advection(Image):
    """TODO"""
    def __init__(self, velocity=None):
        super().__init__()
        self._v = np.empty(shape=self.image_size + [2], dtype=np.float64) if velocity is None else np.array(velocity, dtype=np.float64, order='C')

    def __add__(self, other):
        advection = Advection(velocity=self.v + other.v)
        return advection

    def __sub__(self, other):
        advection = Advection(velocity=self.v - other.v)
        return advection

    def __mul__(self, other):
        assert np.isscalar(other), 'Only scalar * advection multiplication is supported'
        advection = Advection(velocity=self.v * other)
        return advection

    def __neg__(self):
        advection = Advection(velocity=-self.v)
        return advection

    def plot_velocity(self, downscale_factor=0.125):
        """TODO"""
        new_shape = (downscale_factor * np.array(self.image_size)).astype(int)
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
        x = np.linspace(min(self.x.min(), other.x.min()), max(self.x.max(), other.x.max()), self.image_size[0])
        y = np.linspace(min(self.y.min(), other.y.min()), max(self.y.max(), other.y.max()), self.image_size[1])
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
