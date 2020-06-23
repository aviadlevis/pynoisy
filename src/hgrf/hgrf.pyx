cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free

cdef extern from "Python.h":
    char* PyUnicode_AsUTF8(object unicode)

cdef extern from "main.c":
    int c_main_mpi(int argc, char *argv[])
    int c_main(int nk, int ni, int nj, double* source, double* output_video)

def run(nt, nx, ny, np.ndarray[double, ndim=3, mode="c"] source):
    """TODO"""
    cdef np.ndarray[double, ndim=3, mode="c"] output_video = np.zeros(shape=(nt, nx, ny), dtype=np.float64)
    c_main(nt, nx, ny, &source[0,0,0], &output_video[0,0,0])
    return np.asarray(output_video)

def run_mpi(args):
    cdef char **c_argv
    args = [b'mpirun'] + [bytes(x, encoding='utf8') for x in args]
    c_argv = <char**>malloc(sizeof(char*) * len(args))
    for idx, s in enumerate(args):
        c_argv[idx] = PyUnicode_AsUTF8(s)
    c_main_mpi(len(args), c_argv)