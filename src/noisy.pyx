cimport numpy as np
import numpy as np
from cpython cimport array

cdef extern from "noisy.h":
    enum: N

cdef extern from "evolve.c":
    void grid_function_calc(double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4], double v[N][N][4][2],
                            double T[N][N], double *Kmax, double *Vmax, double PARAM_RAT,
                            double principal_angle_image[N][N], double advection_velocity_image[N][N][2],
                            double diffusion_coefficient_image[N][N], double correlation_time_image[N][N])
    void evolve_diffusion(double dell[N][N], double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4], double dt)
    void linear_mc(double x1, double x2, double x3, double *lout, double *rout)
    void reconstruct_lr(double d0, double d1, double d2, double d3, double *d_left, double *d_right)
    double lr_to_flux(double d_left, double d_right, double v)
    void evolve_advection(double dell[N][N], double v[N][N][4][2], double dt)
    void evolve_noise(double dell[N][N], double dt)
    void evolve_decay(double dell[N][N], double T[N][N], double dt)

cdef extern from "model_disk.c":
    double correlation_length(double x,double y, double PARAM_RCH, double PARAM_LAM)
    double correlation_time(double x,double y, double PARAM_TAU, double PARAM_RCH)
    void get_correlation_time_image(double correlation_time_image[N][N], double PARAM_TAU, double PARAM_RCH)
    double diffusion_coefficient(double x,double y, double PARAM_TAU, double PARAM_RCH, double PARAM_LAM)
    void get_diffusion_coefficient(double diffusion_coefficient_image[N][N], double PARAM_TAU, double PARAM_LAM, double PARAM_RCH)
    void principal_axis_func(double x, double y, double *e1x, double *e1y, double *e2x, double *e2y, double opening_angle)
    double W_Keplerian(double x, double y, double direction, double PARAM_RCH)
    void advection_velocity(double x, double y, double va[2], double direction, double PARAM_RCH)
    void get_advection_velocity_image(double velocity[N][N][2], double direction, double PARAM_RCH)
    double phi_func(double x, double y, double opening_angle)
    void principal_angle_image(double angle[N][N], double opening_angle)
    double envelope(double x, double y, double PARAM_RCH)
    void get_envelope_image(double envelope_image[N][N], double PARAM_RCH)
    void noise_model(double del_noise[N][N], double dt, double PARAM_EPS)

cdef extern from "main.c":
    int cmain(double PARAM_RAT, double PARAM_AMP, double PARAM_EPS, double tf,
              double* principal_angle_image, double* advection_velocity_image,
              double* diffusion_coefficient_image, double* correlation_time_image, double* envelope_image)
    void xy_image(double x[N][N], double y[N][N])

cdef extern from "image.c":
    void emit_image(double dell[N][N], int n)
    int compare_doubles(const void *a, const void *b)
    void john_pal(double data, double min, double max, int *pRed, int *pGreen, int *pBlue)


def get_image_size():
    """TODO"""
    return [N, N]

def run_main(PARAM_RAT, PARAM_AMP, PARAM_EPS, evolution_length,
             np.ndarray[double, ndim=2, mode="c"] c_principal_angle_image,
             np.ndarray[double, ndim=3, mode="c"] c_advection_velocity_image,
             np.ndarray[double, ndim=2, mode="c"] c_diffusion_coefficient_image,
             np.ndarray[double, ndim=2, mode="c"] c_correlation_time_image,
             np.ndarray[double, ndim=2, mode="c"] c_envelope_image):
    """TODO"""
    cmain(PARAM_RAT, PARAM_AMP, PARAM_EPS, evolution_length, &c_principal_angle_image[0,0], &c_advection_velocity_image[0,0,0],
          &c_diffusion_coefficient_image[0,0], &c_correlation_time_image[0,0], &c_envelope_image[0,0])


def get_xy_grid():
    """TODO"""
    cdef double x[N][N]
    cdef double y[N][N]
    xy_image(x, y)
    return np.asarray(x), np.asarray(y)

def get_disk_angle(opening_angle):
    """TODO"""
    cdef double angle[N][N]
    principal_angle_image(angle, opening_angle)
    return np.asarray(angle)

def get_disk_correlation_time(PARAM_TAU, PARAM_RCH):
    """TODO"""
    cdef double correlation_time_image[N][N]
    get_correlation_time_image(correlation_time_image, PARAM_TAU, PARAM_RCH)
    return np.asarray(correlation_time_image)

def get_disk_diffusion_coefficient(PARAM_TAU, PARAM_LAM, PARAM_RCH):
    """TODO"""
    cdef double diffusion_coefficient_image[N][N]
    get_diffusion_coefficient(diffusion_coefficient_image, PARAM_TAU, PARAM_LAM, PARAM_RCH)
    return np.asarray(diffusion_coefficient_image)

def get_disk_velocity(direction, PARAM_RCH):
    """TODO"""
    cdef double velocity[N][N][2]
    get_advection_velocity_image(velocity, direction, PARAM_RCH)
    return np.asarray(velocity)

def get_disk_envelope(PARAM_RCH):
    """TODO"""
    cdef double envelope_image[N][N]
    get_envelope_image(envelope_image, PARAM_RCH)
    return np.asarray(envelope_image)






