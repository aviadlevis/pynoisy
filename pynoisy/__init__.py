"""
TODO: Some documentation and general description goes here.
"""
from enum import Enum
from matplotlib import animation
import skimage.transform
import pickle
import os

class RotationDirection(Enum):
    clockwise = -1.0
    counter_clockwise = 1.0


class Image(object):
    """TODO"""
    def __init__(self):
        self._x, self._y = core.get_xy_grid()
        self._image_size = core.get_image_size()
        self._r = np.sqrt(self.x ** 2 + self.y ** 2)

    def shift(self, x0, y0):
        """TODO"""
        self._x += x0
        self._y += y0

    def rotate(self, angle):
        self._data = skimage.transform.rotate(self.data, angle)

    @property
    def image_size(self):
        return self._image_size

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def r(self):
        return self._r


class Movie(object):
    """TODO"""

    def __init__(self, frames=None, duration=None):
        self._frames = frames
        self._duration = duration
        self._image_size = core.get_image_size()

    def __mul__(self, other):
        assert np.isscalar(other), 'Only scalar * Movie multiplication is supported'
        movie = Movie(frames=self.frames * other, duration=self.duration)
        return movie

    def get_animation(self, vmin=None, vmax=None, fps=10, output=None):
        """TODO"""
        # initialization function: plot the background of each frame
        def init():
            im.set_data(np.zeros(self.image_size), vmin=-5, vmax=5)
            return [im]

        # animation function.  This is called sequentially
        def animate(i):
            im.set_array(self.frames[i])
            return [im]

        fig, ax = plt.subplots()
        im = plt.imshow(np.zeros(self.image_size))
        plt.colorbar()
        vmin = self.frames.min() if vmin is None else vmin
        vmax = self.frames.min() if vmax is None else vmax
        im.set_clim(vmin, vmax)
        anim = animation.FuncAnimation(fig, animate, frames=self.num_frames, interval=1e3 / fps)

        if output is not None:
            anim.save(output, writer='imagemagick', fps=fps)
        return anim

    def duplicate_single_frame(self, frame, num_frames):
        self._frames = [frame] * num_frames

    def reverse_time(self):
        return Movie(np.flip(self.frames, axis=0), duration=self.duration)

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
    def image_size(self):
        return self._image_size

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


class MovieSamples(Movie):
    duplicate_single_frame = property()

    def __init__(self, movie_list=None):
        duration = None
        if movie_list is not None:
            duration = [movie.duration for movie in movie_list]
            assert duration.count(duration[0]) == len(duration), 'All movie durations should be identical'
            duration = duration[0]
        super().__init__(duration=duration)
        self._movie_list = movie_list
        self._frames = np.array([movie.frames for movie in movie_list])

    def __mul__(self, other):
        movie_list = [super(MovieSamples, self).__mul__(movie) for movie in self.movie_list]
        return MovieSamples(movie_list)

    def __getitem__(self, item):
        return self.movie_list[item]

    def get_animation(self, vmin=None, vmax=None, fps=10, output=None):
        anim_list = [super(MovieSamples, self).get_animation(vmin, vmax, fps, output) for movie in self.movie_list]
        return anim_list

    def reverse_time(self):
        movie_list = [super(MovieSamples, self).reverse_time(movie) for movie in self.movie_list]
        return MovieSamples(movie_list)

    def mean(self):
        return Movie(self.frames.mean(axis=0), self.duration)

    def std(self):
        return Movie(self.frames.std(axis=0), self.duration)

    @property
    def movie_list(self):
        return self._movie_list


from pynoisy.diffusion import *
from pynoisy.advection import *
from pynoisy.envelope import *
from pynoisy.forward import *
from pynoisy.inverse import *