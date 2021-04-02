import setuptools
from numpy.distutils.core import setup
import numpy
import subprocess
import sys

args = sys.argv[1:]

if 'clean' in args:
    subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.c", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash")
    subprocess.Popen("rm -rf dist", shell=True, executable="/bin/bash")
    subprocess.Popen(["make clean"], shell=True, stdout=subprocess.PIPE, cwd="./inoisy")
    subprocess.Popen(["make distclean"], shell=True, stdout=subprocess.PIPE, cwd="./inoisy")

# Compile general_xy model and create a symbolic link in the executable path
if 'install' in args or 'develop' in args:
    subprocess.Popen(["make matrices"], shell=True, stdout=subprocess.PIPE, cwd="./inoisy")

setup(
    name='pynoisy',
    version='1.5',
    packages = setuptools.find_packages(),
    include_dirs=[numpy.get_include()],
    install_requires=["numpy",
                      "netcdf4",
                      "scipy",
                      "matplotlib",
                      "pandas",
                      "xarray",
                      "tensorboardX",
                      "future"]
)




