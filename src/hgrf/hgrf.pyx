cimport numpy as np
import numpy as np

cdef extern from "main.c":
    int c_main()

def run(argv):
    """TODO"""
    c_main()