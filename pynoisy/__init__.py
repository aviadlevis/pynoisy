"""
This is a python wrapper for a modified version of inoisy which supports arbitrary xarray matrices as diffusion tensor
parameters for generation of Gaussian Random Fields as a solution to a Stochastic Partial Differential Equation [1].
This code is used for inference of stochastic fluid dynamics of black-hole accretion from Event Horizon Telescope (EHT)
measurements.

This code was created by Aviad Levis, California Institute of Technology.
The original inoisy source code[2] was created by by David Daeyoung Lee and Charles Gammie, University of Illinois.

References
----------
.. [1] Lee, D. and Gammie, C.F., 2021. Disks as Inhomogeneous, Anisotropic Gaussian Random Fields.
   The Astrophysical Journal, 906(1), p.39. url: https://iopscience.iop.org/article/10.3847/1538-4357/abc8f3/meta
.. [2] inoisy code: https://github.com/AFD-Illinois/inoisy
"""
import pynoisy.diffusion
import pynoisy.advection
import pynoisy.envelope
import pynoisy.forward
# import pynoisy.inverse
import pynoisy.utils
import pynoisy.linalg
import pynoisy.visualization

# Add inoisy executables directory to path
import os
os.environ['PATH'] = os.environ['INOISY_DIR'] + ':' + os.environ['PATH']

