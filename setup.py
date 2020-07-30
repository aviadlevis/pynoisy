import setuptools
from numpy.distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import subprocess
import sys
import os, tempfile

def symlink(target, link_name, overwrite=True):
    '''
    Create a symbolic link named link_name pointing to target.
    If link_name exists then FileExistsError is raised, unless overwrite=True.
    When trying to overwrite a directory, IsADirectoryError is raised.
    '''

    if not overwrite:
        os.symlink(target, link_name)
        return

    # os.replace() may fail if files are on different filesystems
    link_dir = os.path.dirname(link_name)

    # Create link to target with temporary filename
    while True:
        temp_link_name = tempfile.mktemp(dir=link_dir)

        # os.* functions mimic as closely as possible system functions
        # The POSIX symlink() returns EEXIST if link_name already exists
        # https://pubs.opengroup.org/onlinepubs/9699919799/functions/symlink.html
        try:
            os.symlink(target, temp_link_name)
            break
        except FileExistsError:
            pass

    # Replace link_name with temp_link_name
    try:
        # Pre-empt os.replace on a directory with a nicer message
        if os.path.isdir(link_name):
            raise IsADirectoryError(f"Cannot symlink over existing directory: '{link_name}'")
        os.replace(temp_link_name, link_name)
    except:
        if os.path.islink(temp_link_name):
            os.remove(temp_link_name)
        raise

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

# Compile general_xy model and create a symbolic link in the executable path
subprocess.Popen(["make general_xy"], shell=True, stdout=subprocess.PIPE, cwd="./inoisy")
symlink(os.path.abspath('./inoisy/general_xy'), sys.executable.split('python')[0] + 'general_xy')