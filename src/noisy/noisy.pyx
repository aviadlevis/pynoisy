cimport numpy as np
import numpy as np
from cpython cimport array
from libcpp cimport bool
import os

cdef extern from "evolve.c":
    void get_diffusion_tensor_image(int nx, int ny, double* F_coeff_gradx, double PARAM_RAT, double* principal_angle_image,
                                    double* diffusion_coefficient_image)
    void get_laplacian_image(int nt, int nx, int ny, double* lap, double PARAM_RAT, double* principal_angle_image,
                             double* diffusion_coefficient_image, double* advection_velocity_image,
                             double* correlation_time_image, double* frames)

cdef extern from "model_disk.c":
    void get_correlation_length_image(int nx, int ny, double* correlation_length_image, double PARAM_RCH, double PARAM_LAM)
    void get_correlation_time_image(int nx, int ny, double* correlation_time_image, double PARAM_TAU, double PARAM_RCH)
    void get_diffusion_coefficient(int nx, int ny, double* diffusion_coefficient_image, double PARAM_TAU, double PARAM_LAM, double PARAM_RCH)
    void get_advection_velocity_image(int nx, int ny, double* velocity, double direction, double PARAM_RCH)
    void principal_angle_image(int nx, int ny,  double* angle, double opening_angle)
    void get_envelope_image(int nx, int ny, double* envelope_image, double PARAM_RCH)

cdef extern from "main.c":
    int main_asymmetric(int nt, int nx, int ny, double PARAM_RAT, double PARAM_EPS, double tf,
              double* principal_angle_image, double* advection_velocity_image,
              double* diffusion_coefficient_image, double* correlation_time_image,
              double* output_video, bool verbose, int seed)
    int main_symmetric(int nt, int nx, int ny, double PARAM_RAT, double tf,
              double* principal_angle_image, double* advection_velocity_image,
              double* diffusion_coefficient_image, double* correlation_time_image,
              double* output_video, double* source, bool verbose)
    void xy_image(int nx, int ny, double* x, double* y)


def run_asymmetric(nt, nx, ny, PARAM_RAT, PARAM_EPS, evolution_length,
             np.ndarray[double, ndim=2, mode="c"] c_principal_angle_image,
             np.ndarray[double, ndim=3, mode="c"] c_advection_velocity_image,
             np.ndarray[double, ndim=2, mode="c"] c_diffusion_coefficient_image,
             np.ndarray[double, ndim=2, mode="c"] c_correlation_time_image,
             verbose, seed=0):
    """TODO"""
    cdef np.ndarray[double, ndim=3, mode="c"] output_video = np.zeros(shape=(nt, nx, ny), dtype=np.float64)
    main_asymmetric(nt, nx, ny, PARAM_RAT, PARAM_EPS, evolution_length, &c_principal_angle_image[0,0],
          &c_advection_velocity_image[0,0,0], &c_diffusion_coefficient_image[0,0], &c_correlation_time_image[0,0],
          &output_video[0,0,0], verbose, seed)
    return np.asarray(output_video)


def run_symmetric(nt, nx, ny, PARAM_RAT, evolution_length,
             np.ndarray[double, ndim=2, mode="c"] c_principal_angle_image,
             np.ndarray[double, ndim=3, mode="c"] c_advection_velocity_image,
             np.ndarray[double, ndim=2, mode="c"] c_diffusion_coefficient_image,
             np.ndarray[double, ndim=2, mode="c"] c_correlation_time_image,
             np.ndarray[double, ndim=3, mode="c"] source,
             verbose):
    """TODO"""
    cdef np.ndarray[double, ndim=3, mode="c"] output_video = np.zeros(shape=(nt, nx, ny), dtype=np.float64)
    main_symmetric(nt, nx, ny, PARAM_RAT, evolution_length, &c_principal_angle_image[0,0],
          &c_advection_velocity_image[0,0,0], &c_diffusion_coefficient_image[0,0], &c_correlation_time_image[0,0],
          &output_video[0,0,0], &source[0,0,0], verbose)
    return np.asarray(output_video)


def get_xy_grid(nx, ny):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] x = np.zeros(shape=(nx, ny), dtype=np.float64)
    cdef np.ndarray[double, ndim=2, mode="c"] y = np.zeros(shape=(nx, ny), dtype=np.float64)
    xy_image(nx, ny, &x[0,0], &y[0,0])
    return np.asarray(x), np.asarray(y)

def get_disk_angle(nx, ny, opening_angle):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] angle = np.zeros(shape=(nx, ny), dtype=np.float64)
    principal_angle_image(nx, ny, &angle[0,0], opening_angle)
    return np.asarray(angle)

def get_disk_correlation_length(nx, ny, PARAM_RCH, PARAM_LAM):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] correlation_length_image = np.zeros(shape=(nx, ny), dtype=np.float64)
    get_correlation_length_image(nx, ny, &correlation_length_image[0,0], PARAM_RCH, PARAM_LAM)
    return np.asarray(correlation_length_image)

def get_disk_correlation_time(nx, ny, PARAM_TAU, PARAM_RCH):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] correlation_time_image = np.zeros(shape=(nx, ny), dtype=np.float64)
    get_correlation_time_image(nx, ny, &correlation_time_image[0,0], PARAM_TAU, PARAM_RCH)
    return np.asarray(correlation_time_image)

def get_disk_diffusion_coefficient(nx, ny, PARAM_TAU, PARAM_LAM, PARAM_RCH):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] diffusion_coefficient_image = np.zeros(shape=(nx, ny), dtype=np.float64)
    get_diffusion_coefficient(nx, ny, &diffusion_coefficient_image[0,0], PARAM_TAU, PARAM_LAM, PARAM_RCH)
    return np.asarray(diffusion_coefficient_image)

def get_disk_velocity(nx, ny, direction, PARAM_RCH):
    """TODO"""
    cdef np.ndarray[double, ndim=3, mode="c"] velocity = np.zeros(shape=(nx, ny, 2), dtype=np.float64)
    get_advection_velocity_image(nx, ny, &velocity[0,0,0], direction, PARAM_RCH)
    v = np.asarray(velocity)
    return v[...,0], v[...,1]

def get_disk_envelope(nx, ny, PARAM_RCH):
    """TODO"""
    cdef np.ndarray[double, ndim=2, mode="c"] envelope_image = np.zeros(shape=(nx, ny), dtype=np.float64)
    get_envelope_image(nx, ny, &envelope_image[0,0], PARAM_RCH)
    return np.asarray(envelope_image)

def get_diffusion_tensor(nx, ny, PARAM_RAT,
                         np.ndarray[double, ndim=2, mode="c"] c_principal_angle_image,
                         np.ndarray[double, ndim=2, mode="c"] c_diffusion_coefficient_image):

    cdef np.ndarray[double, ndim=4, mode="c"] diffusion_tensor = np.zeros(shape=(nx, ny, 4, 2), dtype=np.float64)
    get_diffusion_tensor_image(nx, ny, &diffusion_tensor[0,0,0,0], PARAM_RAT, &c_principal_angle_image[0,0], &c_diffusion_coefficient_image[0,0])
    return np.asarray(diffusion_tensor)

def get_laplacian(nt, nx, ny, PARAM_RAT,
                  np.ndarray[double, ndim=2, mode="c"] c_principal_angle_image,
                  np.ndarray[double, ndim=2, mode="c"] c_diffusion_coefficient_image,
                  np.ndarray[double, ndim=3, mode="c"] c_advection_velocity_image,
                  np.ndarray[double, ndim=2, mode="c"] c_correlation_time_image,
                  np.ndarray[double, ndim=3, mode="c"] frames):
    """TODO"""
    cdef np.ndarray[double, ndim=3, mode="c"] lap = np.zeros(shape=(nt, nx, ny), dtype=np.float64)
    get_laplacian_image(nt, nx, ny, &lap[0,0,0], PARAM_RAT, &c_principal_angle_image[0,0], &c_diffusion_coefficient_image[0,0],
                        &c_advection_velocity_image[0,0,0], &c_correlation_time_image[0,0], &frames[0,0,0])
    return np.asarray(lap)

