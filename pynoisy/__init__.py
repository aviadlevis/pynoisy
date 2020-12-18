"""
TODO: Some documentation and general description goes here.
"""
import pynoisy.diffusion
import pynoisy.advection
import pynoisy.envelope
import pynoisy.forward
import pynoisy.inverse
import pynoisy.utils

import os
import pathlib
current_path = pathlib.Path(__file__).parent.absolute()
os.environ['PATH'] = os.path.join(current_path.parent, 'inoisy') + ':' + os.environ['PATH']




