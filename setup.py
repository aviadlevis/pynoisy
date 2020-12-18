import setuptools
from numpy.distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import subprocess
import sys

args = sys.argv[1:]

if 'cleanall' in args:
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.c", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash")
    subprocess.Popen(["make distclean"], shell=True, stdout=subprocess.PIPE, cwd="./inoisy")

    # Now do a normal clean
    sys.argv[1] = "clean"

# Compile general_xy model and create a symbolic link in the executable path
if 'install' in args or 'develop' in args:
    subprocess.Popen(["make matrices"], shell=True, stdout=subprocess.PIPE, cwd="./inoisy")

extensions = [
    Extension(
        'noisy_core',
        sources=['src/noisy.pyx'],
        libraries=['gsl', 'blas'],
    ),
    Extension(
        'hgrf_core',
        sources=['src/hgrf.pyx'],
        libraries=['gsl', 'blas'],
    ),
]

setup(
    name='pynoisy',
    version='1.5',
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




