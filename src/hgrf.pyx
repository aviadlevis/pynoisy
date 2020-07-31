cimport numpy as np
import numpy as np

cdef extern from "general_xy_matrices.c":
    void get_correlation_time_image(int ni, int nj, double* correlation_time_image, double param_tau, double param_rct)
    void get_correlation_length_image(int ni, int nj, double* correlation_length_image, double param_rct, double param_lam)
    void get_spatial_angle_image(int ni, int nj, double* spatial_angle_image, double param_rct, double param_theta)
    void get_velocity_image(int ni, int nj, double* spatio_temporal_angle, double direction, double param_rct)


def get_generalxy_correlation_time(nx, ny, param_tau, param_rct):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] correlation_time_image = np.zeros(shape=(nx, ny), dtype=np.float64)
    get_correlation_time_image(nx, ny, &correlation_time_image[0,0], param_tau, param_rct)
    return np.asarray(correlation_time_image).T

def get_generalxy_correlation_length(nx, ny, param_rct, param_lam):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] correlation_length_image = np.zeros(shape=(nx, ny), dtype=np.float64)
    get_correlation_length_image(nx, ny, &correlation_length_image[0,0], param_rct, param_lam)
    return np.asarray(correlation_length_image).T

def get_generalxy_spatial_angle(nx, ny, param_rct, opening_angle):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] spatial_angle_image = np.zeros(shape=(nx, ny), dtype=np.float64)
    get_spatial_angle_image(nx, ny, &spatial_angle_image[0,0], param_rct, opening_angle)
    return np.asarray(spatial_angle_image).T

def get_generalxy_velocity(nx, ny, direction, param_rct):
    """TODO"""
    cdef np.ndarray[double, ndim=3, mode="c"] velocity = np.zeros(shape=(nx, ny, 2), dtype=np.float64)
    get_velocity_image(nx, ny, &velocity[0,0,0], direction, param_rct)
    v = np.asarray(velocity)
    return v[...,0].T, v[...,1].T
