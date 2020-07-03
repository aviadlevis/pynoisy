import setuptools
from numpy.distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import os


extensions = [
    Extension(
        'noisy_core', ['src/noisy/noisy.pyx'],
        include_dirs=[numpy.get_include()],
        libraries=['gsl', 'blas'],
        library_dirs=['/usr/lib/x86_64-linux-gnu/'],
        depends=['src/noisy/noisy.h', 'src/noisy/evolve.c', 'src/noisy/image.c',
                 'src/noisy/model_disk.c', 'src/noisy/model_uniform.c']
    ),
    Extension(
        'hgrf_core',
        sources=['src/hgrf/hgrf.pyx', 'src/hgrf/hdf5_utils.c'],
        include_dirs=[numpy.get_include(), '/home/aviad/Code/hypre/src/hypre/include/',
                      '/usr/lib/x86_64-linux-gnu/openmpi/include/', '/home/aviad/anaconda3/envs/eht/include/'],
        libraries=['gsl', 'blas', 'mpi', 'hdf5', 'HYPRE'],
        library_dirs=['/usr/lib/x86_64-linux-gnu/openmpi/lib/', '/home/aviad/Code/hypre/src/hypre/lib/'],
    )
]

setup(
    name='pynoisy',
    version='1.1',
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
