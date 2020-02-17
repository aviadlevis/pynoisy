cdef extern from "model_disk.c":
    double correlation_length(double x,double y)
    double correlation_time(double x,double y)
    double diffusion_coefficient(double x,double y)
    void principal_axis_func(double x, double y, double *e1x, double *e1y, double *e2x, double *e2y)
    double W_Keplerian(double x, double y)
    void advection_velocity(double x, double y, double va[2])
    double phi_func(double x, double y)
    double envelope(double x, double y)
    void noise_model(double del_noise[256][256], double dt)

cdef extern from "evolve.c":
    void grid_function_calc(double F_coeff_gradx[256][256][4], double F_coeff_grady[256][256][4], double v[256][256][4][2], double T[256][256], double *Kmax, double *Vmax)
    void evolve_diffusion(double dell[256][256], double F_coeff_gradx[256][256][4], double F_coeff_grady[256][256][4], double dt)
    void linear_mc(double x1, double x2, double x3, double *lout, double *rout)
    void reconstruct_lr(double d0, double d1, double d2, double d3, double *d_left, double *d_right)
    double lr_to_flux(double d_left, double d_right, double v)
    void evolve_advection(double dell[256][256], double v[256][256][4][2], double dt)
    void evolve_noise(double dell[256][256], double dt)
    void evolve_decay(double dell[256][256], double T[256][256], double dt)


cdef extern from "image.c":
    void emit_image(double dell[256][256], int n)
    int compare_doubles(const void *a, const void *b)
    void john_pal(double data, double min, double max, int *pRed, int *pGreen, int *pBlue)

cdef extern from "main.c":
    int cmain()

def run_main() -> int:
    cmain()