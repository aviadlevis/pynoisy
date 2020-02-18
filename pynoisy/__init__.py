"""
TODO: Some documentation and general description goes here.
"""

import core
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

class AdvectionVelocity(object):
    """TODO"""
    def __init__(self):
        self._v = np.empty(shape=core.get_image_size() + [2], dtype=np.float64)
        self._x, self._y = core.get_xy_grid()

    def quiver(self, downscale_factor=0.125):
        """TODO"""
        scaled_v = scipy.ndimage.zoom(self.v, (downscale_factor, downscale_factor, 1))
        scaled_x = scipy.ndimage.zoom(self.x, downscale_factor)
        scaled_y = scipy.ndimage.zoom(self.y, downscale_factor)
        plt.quiver(scaled_x, scaled_y, scaled_v[..., 0], scaled_v[..., 1])
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

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


class KeplerAdvectionVelocity(AdvectionVelocity):
    """TODO"""
    def __init__(self):
        super().__init__()
        self._v = core.get_kepler_velocity()




