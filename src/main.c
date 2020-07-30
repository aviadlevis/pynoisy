
/*

noisy:

Generate a gaussian random field in 2D with 
locally anisotropic correlation function, 
locally varying correlation time.

Follows the technique of 
Lindgren, Rue, and Lindstr\:om 2011, J.R. Statist. Soc. B 73, pp 423-498.
https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2011.00777.x
in particular, implements eq. 17, which has power spectrum given by eq. 18.

Based on work by 
David Daeyoung Lee 
Charles Gammie
on applications in disk turbulence.

CFG 22 Dec 2019

*/

#include "noisy.h"

int main_asymmetric(
    int nt, int nx, int ny,
    double PARAM_RAT,
    double PARAM_EPS,
    double tf,
    double* principal_angle_image,
    double* advection_velocity_image,
    double* diffusion_coefficient_image,
    double* correlation_time_image,
    double* output_video,
    bool verbose,
    int seed
    )
{
    void grid_function_calc(int nx, int ny, double F_coeff_gradx[nx][ny][4], double F_coeff_grady[nx][ny][4],
        double v[nx][ny][4][2], double T[nx][ny], double *Kmax, double *Vmax,
        double PARAM_RAT, double* principal_angle_image, double* advection_velocity_image,
        double* diffusion_coefficient_image, double* correlation_time_image);
    void evolve_diffusion(int nx, int ny, double _del[nx][ny], double F_coeff_gradx[nx][ny][4], double F_coeff_grady[nx][ny][4],
        double dt);
    void evolve_advection(int nx, int ny, double _del[nx][ny], double v[nx][ny][4][2], double dt);
    void evolve_decay(int nx, int ny, double _del[nx][ny], double T[nx][ny], double dt);
    void evolve_noise(int nx, int ny, double del[nx][ny], double dt, double PARAM_EPS, gsl_rng* r);

    double dx = 1.0/nx;
    double dy = 1.0/ny;

    /* correlation length l = sqrt(K*T) */
    /* correlation time is t = T */
    /* so diffusion coefficient is l^2/t */

    int i,j ;
    double Dtl = tf / (double) nt;   /* image file output cadence */

    /* calculate some grid functions */
    double _del[nx][ny];
    double T[nx][ny];
    double v[nx][ny][4][2];
    double F_coeff_gradx[nx][ny][4];
    double F_coeff_grady[nx][ny][4];
    memset(F_coeff_gradx, 0, sizeof(double) * nx * ny);
    memset(F_coeff_grady, 0, sizeof(double) * nx * ny);

    double Kmax = 0.;
    double Vmax = 0.;


    grid_function_calc(nx, ny, F_coeff_gradx, F_coeff_grady, v, T, &Kmax, &Vmax,
        PARAM_RAT, principal_angle_image, advection_velocity_image,
        diffusion_coefficient_image, correlation_time_image);

    if (verbose) {
        fprintf(stderr,"Vmax: %g\n",Vmax);
        fprintf(stderr,"Kmax: %g\n",Kmax);
    }

    /* now that we know Kmax and Vmax, set timestep */
    double d = fmin(dx,dy);
    /* courant-limited timestep for diffusive part.  cour = 1 is
       the rigorous stability limit if dx = dy, RAT = 1 */
    double cour = 0.45;
    double dtdiff = cour*0.25*d*d/Kmax;
    double dtadv = cour*0.5*d/Vmax;
    double dt = fmin(dtdiff, dtadv);
    if (verbose) {
        fprintf(stderr,"dt,dtdiff,dtadv: %g %g %g\n",dt, dtdiff, dtadv);
    }

    /* initial conditions (typically zero) */
    /*
    double sigsq = 0.1*0.1 ;
    double x,y;
    */

    /* Initiialize rng */
    gsl_rng* r;
    r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, seed);
    if (verbose) {
        fprintf(stderr,"seed = %d \n", seed);
    }

    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        _del[i][j] = 0.;
    }


    int n = 0;
    int nstep = 0;
    double rms = 0.;
    double t = 0.;
    double tl = Dtl;
    tf += Dtl;
    while(t < tf){

         /* operator split */
        evolve_noise(nx, ny, _del, dt, PARAM_EPS, r);
        evolve_diffusion(nx, ny, _del, F_coeff_gradx, F_coeff_grady, dt);
        evolve_advection(nx, ny, _del, v, dt);
        evolve_decay(nx, ny, _del, T, dt);

        /* periodically execute diagnostics */
        if(t > tl) {
            /* check rms and output  random field */
            rms = 0.;
            for(i=0;i<nx;i++)
            for(j=0;j<ny;j++) {
                rms += _del[i][j]*_del[i][j];
                output_video[n] = _del[i][j];
                n++;
            }
            rms = sqrt(rms)/sqrt(nx*ny);

            if (verbose) {
                fprintf(stderr,"%lf %lf\n",t, rms);
            }
            /* set time for next diagnostic output */
            tl += Dtl;
        }
        nstep++;
        t += dt;
    }

    /* Free GSL rng state */
    gsl_rng_free(r);
}


int main_symmetric(
    int nt, int nx, int ny,
    double PARAM_RAT,
    double tf,
    double* principal_angle_image,
    double* advection_velocity_image,
    double* diffusion_coefficient_image,
    double* correlation_time_image,
    double* output_video,
    double* source,
    bool verbose
    )
{
    void grid_function_calc(int nx, int ny, double F_coeff_gradx[nx][ny][4], double F_coeff_grady[nx][ny][4],
        double v[nx][ny][4][2], double T[nx][ny], double *Kmax, double *Vmax,
        double PARAM_RAT, double* principal_angle_image, double* advection_velocity_image,
        double* diffusion_coefficient_image, double* correlation_time_image);
    void evolve_diffusion(int nx, int ny, double _del[nx][ny], double F_coeff_gradx[nx][ny][4], double F_coeff_grady[nx][ny][4],
        double dt);
    void evolve_advection(int nx, int ny, double _del[nx][ny], double v[nx][ny][4][2], double dt);
    void evolve_decay(int nx, int ny, double _del[nx][ny], double T[nx][ny], double dt);
    void evolve_source(int nx, int ny, double del[nx][ny], double dt, double* source);

    double dx = 1.0/nx;
    double dy = 1.0/ny;

    /* correlation length l = sqrt(K*T) */
    /* correlation time is t = T */
    /* so diffusion coefficient is l^2/t */

    int i,j ;
    double Dtl = tf / (double) nt;   /* image file output cadence */

    /* calculate some grid functions */
    double _del[nx][ny];
    double T[nx][ny];
    double v[nx][ny][4][2];
    double F_coeff_gradx[nx][ny][4];
    double F_coeff_grady[nx][ny][4];
    memset(F_coeff_gradx, 0, sizeof(double) * nx * ny);
    memset(F_coeff_grady, 0, sizeof(double) * nx * ny);
    double Kmax = 0.;
    double Vmax = 0.;

    grid_function_calc(nx, ny, F_coeff_gradx, F_coeff_grady, v, T, &Kmax, &Vmax,
        PARAM_RAT, principal_angle_image, advection_velocity_image,
        diffusion_coefficient_image, correlation_time_image);

    if (verbose) {
        fprintf(stderr,"Vmax: %g\n",Vmax);
        fprintf(stderr,"Kmax: %g\n",Kmax);
    }

    /* now that we know Kmax and Vmax, set timestep */
    double d = fmin(dx,dy);
    /* courant-limited timestep for diffusive part.  cour = 1 is
       the rigorous stability limit if dx = dy, RAT = 1 */
    double cour = 0.45;
    double dtdiff = cour*0.25*d*d/Kmax;
    double dtadv = cour*0.5*d/Vmax;
    double dt = fmin(dtdiff, dtadv);
    if (verbose) {
        fprintf(stderr,"dt,dtdiff,dtadv: %g %g %g\n",dt, dtdiff, dtadv);
    }

    /* initial conditions (typically zero) */
    /*
    double sigsq = 0.1*0.1 ;
    double x,y;
    */
    gsl_rng* r;
    r = gsl_rng_alloc(gsl_rng_mt19937); /* Mersenne twister */
    gsl_rng_set(r, 0);
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        _del[i][j] = 0.;
    }


    int n = 0;
    int nstep = 0;
    double rms = 0.;
    double t = 0.;
    double tl = Dtl;
    while(t <= tf){

         /* operator split */
        evolve_source(nx, ny, _del, dt / Dtl, &source[n]);
        evolve_diffusion(nx, ny, _del, F_coeff_gradx, F_coeff_grady, dt);
        evolve_advection(nx, ny, _del, v, dt);
        evolve_decay(nx, ny, _del, T, dt);

        /* periodically execute diagnostics */
        if(t > tl) {
            /* check rms and output  random field */
            rms = 0.;
            for(i=0;i<nx;i++)
            for(j=0;j<ny;j++) {
                rms += _del[i][j]*_del[i][j] * Dtl;
                output_video[n] = _del[i][j];
                n++;
            }
            rms = sqrt(rms)/sqrt(nx*ny);

            if (verbose) {
                fprintf(stderr,"%lf %lf\n",t, rms);
            }
            /* set time for next diagnostic output */
            tl += Dtl;
        }
        nstep++;
        t += dt;
    }

    if(t > tl) {
        /* check rms and output  random field */
        rms = 0.;
        for(i=0;i<nx;i++)
        for(j=0;j<ny;j++) {
            rms += _del[i][j]*_del[i][j] * Dtl;
            output_video[n] = _del[i][j];
            n++;
        }
        rms = sqrt(rms)/sqrt(nx*ny);

        if (verbose) {
            fprintf(stderr,"%lf %lf\n",t, rms);
        }
    }
}



/* return the coordinates of a zone center */

void ij_to_xy(int nx, int ny, int i, int j, double *x, double *y)
{
    double dx = 1.0/nx;
    double dy = 1.0/ny;

    *x = (i - nx/2)*dx ;
    *y = (j - ny/2)*dy ;
}

/* return image coordinates */
void xy_image(int nx, int ny, double* x, double* y)
{
    int i,j;
    void ij_to_xy(int nx, int ny, int i, int j, double *x, double *y);

    int k = 0;
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        ij_to_xy(nx,ny,i,j, &x[k],&y[k]);
        k++;
    }

}

/* transform random field into fake image */
void apply_envelope(int nx, int ny, double* _del, double* fake_image, double* envelope_image, double PARAM_AMP)
{
    int i,j;
    int k=0;
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        fake_image[k] = envelope_image[k]*exp(-PARAM_AMP*_del[k]);
        k++;
    }
}


