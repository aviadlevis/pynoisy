"""
TODO: Some documentation and general description goes here.
"""
import core
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import interpolate
import ehtim as eh
from pynoisy import Image, Movie


class Envelope(Image):
    """
    The envelope function multiplies the random fields to create synthetic images.
    The equation for each field is: image = exp(-amplitude*del)*envelope
    """
    def __init__(self, data=None):
        super().__init__()
        self._data = np.ones(shape=self.image_size, dtype=np.float64, order='C') if data is None else data

    def __add__(self, other):
        x = np.linspace(min(self.x.min(), other.x.min()), max(self.x.max(), other.x.max()), self.image_size[0])
        y = np.linspace(min(self.y.min(), other.y.min()), max(self.y.max(), other.y.max()), self.image_size[1])
        xx, yy = np.meshgrid(x, y, indexing='ij')
        f1 = interpolate.interp2d(self.x[:,0], self.y[0], self.data, bounds_error=False, fill_value = 0.0)
        f2 = interpolate.interp2d(other.x[:,0], other.y[0], other.data, bounds_error=False, fill_value = 0.0)
        envelope = Envelope(np.array(f1(x, y) + f2(x, y), dtype=np.float64, order='C'))
        envelope._x = xx
        envelope._y = yy
        return envelope

    def __mul__(self, other):
        return Envelope(np.array(self.data * other, dtype=np.float64, order='C'))


    def __truediv__(self, other):
        return Envelope(np.array(self.data / other, dtype=np.float64, order='C'))

    def load_fits(self, path):
        image = eh.image.load_fits(path)
        image = image.regrid_image(image.fovx(), self.image_size[0])
        return Envelope(data=image.imarr())

    def apply(self, movie, amplitude=0.05):
        """
        Parameters
        ----------
        amplitude: float, default = 0.05
            strength of perturbation; image = exp(-amplitude*del)*envelope
        """
        frames = self.data * np.exp(-amplitude * movie.frames)
        return Movie(frames, movie.duration)

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


class DynamicEnvelope(Movie):
    """TODO"""

    def __init__(self, frames=None, duration=None):
        super().__init__(frames, duration)

    def apply(self, movie, amplitude=0.05):
        """
        Parameters
        ----------
        amplitude: float, default = 0.05
            strength of perturbation; image = exp(-amplitude*del)*envelope
        """
        frames = self.frames * np.exp(-amplitude * movie.frames)
        return Movie(frames, movie.duration)

