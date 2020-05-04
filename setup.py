import setuptools
from numpy.distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        'core', ['src/noisy.pyx'],
        include_dirs=['src/', numpy.get_include()],
        libraries=['gsl', 'blas'],
        library_dirs=['/usr/lib/x86_64-linux-gnu/'],
        depends=['src/noisy.h', 'src/evolve.c', 'src/image.c', 'src/model_disk.c', 'src/model_uniform.c']
    )
]

setup(
    name='pynoisy',
    version='1.0',
    packages = setuptools.find_packages(),
    include_dirs=[numpy.get_include()],
    install_requires=["numpy",
                      "scipy",
                      "matplotlib",
                      "pandas",
                      "xarray",
                      "tensorboardX",
                      "future"],
    ext_modules=cythonize(extensions)
)
