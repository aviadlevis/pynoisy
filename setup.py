import setuptools
from numpy.distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import subprocess


extensions = [
    Extension(
        'noisy_core', ['src/noisy/noisy.pyx'],
        include_dirs=[numpy.get_include()],
        libraries=['gsl', 'blas'],
        library_dirs=['/usr/lib/x86_64-linux-gnu/'],
        depends=['src/noisy/noisy.h', 'src/noisy/evolve.c', 'src/noisy/image.c',
                 'src/noisy/model_disk.c', 'src/noisy/model_uniform.c']
    ),
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

subprocess.Popen(["make general_xy"], shell=True, stdout=subprocess.PIPE, cwd="./inoisy")