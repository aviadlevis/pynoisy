import setuptools
from numpy.distutils.core import setup
from setuptools.command.install import install
from distutils.command.clean import clean
from setuptools.command.develop import develop
import numpy
import subprocess

class DevelopCommand(develop):
    def run(self):
        subprocess.Popen(["make matrices"], shell=True, cwd="./inoisy")
        super().run()

class InstallCommand(install):
    def run(self):
        subprocess.Popen(["make matrices"], shell=True, cwd="./inoisy")
        super().run()

class CleanCommand(clean):
    def run(self):
        super().run()
        subprocess.Popen("rm -rf build", shell=True, executable="/bin/bash")
        subprocess.Popen("rm -rf *.c", shell=True, executable="/bin/bash")
        subprocess.Popen("rm -rf *.so", shell=True, executable="/bin/bash")
        subprocess.Popen("rm -rf dist", shell=True, executable="/bin/bash")
        subprocess.Popen(["make clean"], shell=True, cwd="./inoisy")
        subprocess.Popen(["make distclean"], shell=True, cwd="./inoisy")

if __name__ == '__main__':
    setup(
        name='pynoisy',
        version='1.5',
        cmdclass={'install': InstallCommand,
                  'develop': DevelopCommand,
                  'clean': CleanCommand},
        packages = setuptools.find_packages(),
        include_dirs=[numpy.get_include()],
        zip_safe=True
    )



