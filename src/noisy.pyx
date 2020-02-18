import numpy as np
cimport numpy as np

cdef extern from "noisy.h":
    enum: N

cdef extern from "model_disk.c":
    double correlation_length(double x,double y, PARAM_RCH, PARAM_LAM)
    double correlation_time(double x,double y, direction, PARAM_RCH, PARAM_TAU)
    double diffusion_coefficient(double x,double y, double PARAM_RCH, double PARAM_LAM, double PARAM_TAU, double direction)
    void principal_axis_func(double x, double y, double *e1x, double *e1y, double *e2x, double *e2y, double opening_angle)
    double W_Keplerian(double x, double y, double direction, double PARAM_RCH)
    void advection_velocity(double x, double y, double va[2], double direction, double PARAM_RCH, double PARAM_FOV)
    void advection_velocity_image(double velocity[N][N][2], double direction, double PARAM_RCH, double PARAM_FOV)
    double phi_func(double x, double y, double opening_angle)
    double envelope(double x, double y, double PARAM_RCH)
    void noise_model(double del_noise[N][N], double dt, double PARAM_EPS)

cdef extern from "evolve.c":
    void grid_function_calc(double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4], double v[N][N][4][2], double T[N][N], double *Kmax, double *Vmax, double opening_angle, double direction, double PARAM_RCH, double PARAM_FOV, double PARAM_LAM, double PARAM_TAU)
    void evolve_diffusion(double dell[N][N], double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4], double dt, double PARAM_FOV)
    void linear_mc(double x1, double x2, double x3, double *lout, double *rout)
    void reconstruct_lr(double d0, double d1, double d2, double d3, double *d_left, double *d_right)
    double lr_to_flux(double d_left, double d_right, double v)
    void evolve_advection(double dell[N][N], double v[N][N][4][2], double dt, double PARAM_FOV)
    void evolve_noise(double dell[N][N], double dt)
    void evolve_decay(double dell[N][N], double T[N][N], double dt)


cdef extern from "image.c":
    void emit_image(double dell[N][N], int n)
    int compare_doubles(const void *a, const void *b)
    void john_pal(double data, double min, double max, int *pRed, int *pGreen, int *pBlue)

cdef extern from "main.c":
    int cmain(double opening_angle, double direction, double PARAM_RCH, double PARAM_FOV, double PARAM_LAM, double PARAM_TAU, double PARAM_RAT, double PARAM_AMP, double PARAM_EPS)
    void xy_image(double x[N][N], double y[N][N], double PARAM_FOV)

def get_image_size():
    return [N, N]

def run_main(opening_angle, direction, PARAM_RCH, PARAM_FOV, PARAM_LAM, PARAM_TAU, PARAM_RAT, PARAM_AMP, PARAM_EPS):
    cmain(opening_angle, direction, PARAM_RCH, PARAM_FOV, PARAM_LAM, PARAM_TAU, PARAM_RAT, PARAM_AMP, PARAM_EPS)

def get_xy_grid(PARAM_FOV):
    cdef double x[N][N]
    cdef double y[N][N]
    xy_image(x, y, PARAM_FOV)
    return np.asarray(x), np.asarray(y)

def get_kepler_velocity(direction, PARAM_RCH, PARAM_FOV):
    cdef double velocity[N][N][2]
    advection_velocity_image(velocity, direction, PARAM_RCH, PARAM_FOV)
    return np.asarray(velocity)






