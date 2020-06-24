cimport numpy as np
import numpy as np

cdef extern from "param_general_xy.c":
    void get_correlation_time_image(int ni, int nj, double* correlation_time_image, double param_tau, double param_rct)
    void get_correlation_length_image(int ni, int nj, double* correlation_length_image, double param_rct, double param_lam)

cdef extern from "main.c":
    void c_init_mpi()
    void c_end_mpi()
    int c_main(int nk, int ni, int nj, int solver_id, int maxiter, int verbose, double* source, double* output_video,
                double param_tau, double param_lam, double param_rct, double param_r12, double param_r02)

def init_mpi():
    c_init_mpi()

def end_mpi():
    c_end_mpi()

def get_generalxy_correlation_time(nx, ny, param_tau, param_rct):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] correlation_time_image = np.zeros(shape=(nx, ny), dtype=np.float64)
    get_correlation_time_image(nx, ny, &correlation_time_image[0,0], param_tau, param_rct)
    return np.asarray(correlation_time_image)

def get_generalxy_correlation_length(nx, ny, param_rct, param_lam):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] correlation_length_image = np.zeros(shape=(nx, ny), dtype=np.float64)
    get_correlation_length_image(nx, ny, &correlation_length_image[0,0], param_rct, param_lam)
    return np.asarray(correlation_length_image)


def run(nt, nx, ny, solver_id, maxiter, verbose, np.ndarray[double, ndim=3, mode="c"] source,
       param_tau, param_lam, param_rct, param_r12, param_r02):
    """TODO"""
    cdef np.ndarray[double, ndim=3, mode="c"] output_video = np.zeros(shape=(nt, nx, ny), dtype=np.float64)
    c_main(nt, nx, ny, solver_id, maxiter, verbose, &source[0,0,0], &output_video[0,0,0],
            param_tau, param_lam, param_rct, param_r12, param_r02)
    return np.asarray(output_video)
