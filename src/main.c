
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

int cmain(
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
    static double _del[N][N];

    void grid_function_calc(double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4],
        double v[N][N][4][2], double T[N][N], double *Kmax, double *Vmax,
        double PARAM_RAT, double* principal_angle_image, double* advection_velocity_image,
        double* diffusion_coefficient_image, double* correlation_time_image);
    void evolve_diffusion(double _del[N][N], double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4],
        double dt);
    void evolve_advection(double _del[N][N], double v[N][N][4][2], double dt);
    void evolve_decay(double _del[N][N], double T[N][N], double dt);
    void evolve_noise(double del[N][N], double dt, double PARAM_EPS, int seed);

    double dx = PARAM_FOV/N;
    double dy = PARAM_FOV/N;

    /* correlation length l = sqrt(K*T) */
    /* correlation time is t = T */
    /* so diffusion coefficient is l^2/t */

    int i,j ;
    double Dtl = tf / (double) NUM_IMAGES;   /* image file output cadence */

    /* calculate some grid functions */
    static double v[N][N][4][2];
    double Kmax = 0.;
    double Vmax = 0.;
    static double T[N][N];
    double F_coeff_gradx[N][N][4] = {{{0.}}};
    double F_coeff_grady[N][N][4] = {{{0.}}};


    grid_function_calc(F_coeff_gradx, F_coeff_grady, v, T, &Kmax, &Vmax,
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
    gsl_rng * r;
    r = gsl_rng_alloc(gsl_rng_mt19937); /* Mersenne twister */
    gsl_rng_set(r, 0);
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        /*
        ij_to_xy(i,j,&x,&y);
        _del[i][j] = exp(-0.5*(x*x + y*y)/sigsq)/(2.*M_PI*sigsq) ;
        _del[i][j] = PARAM_EPS*gsl_ran_gaussian_ziggurat(r,1.0);
        */
        _del[i][j] = 0.;
    }
    /*
    double delavg;
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        ip = (i+N+1)%N ;
        im = (i+N-1)%N ;
        jp = (j+N+1)%N ;
        jm = (j+N-1)%N ;
        delavg = 0.25*(_del[ip][j] + _del[i][jp] + _del[im][j] + _del[i][jm]);
        ddel[i][j] = delavg;
    }
    for(i=0;i<N;i++)
    for(j=0;j<N;j++)
        _del[i][j] = ddel[i][j];
    */


    int n = 0;
    int nstep = 0;
    double rms = 0.;
    double t = 0.;
    double tl = Dtl;
    tf += Dtl;
    while(t < tf){

         /* operator split */
        evolve_noise(_del, dt, PARAM_EPS, seed);
        evolve_diffusion(_del, F_coeff_gradx, F_coeff_grady, dt);
        evolve_advection(_del, v, dt);
        evolve_decay(_del, T, dt);

        /* periodically execute diagnostics */
        if(t > tl) {
            /* check rms and output  random field */
            rms = 0.;
            for(i=0;i<N;i++)
            for(j=0;j<N;j++) {
                rms += _del[i][j]*_del[i][j];
                output_video[n] = _del[i][j];
                n++;
            }
            rms = sqrt(rms)/N;

            if (verbose) {
                fprintf(stderr,"%lf %lf\n",t, rms);
            }
            /* set time for next diagnostic output */
            tl += Dtl;
        }
        nstep++;
        t += dt;
    }
    /*
    FILE *fp = fopen("noisy.out", "w");
    if(fp == NULL) exit(1);
    for(i=0;i<N;i++)
    for(j=0;j<N;j++)
        fprintf(fp,"%d %d %lf\n",i,j,_del[i][j]);
    */
}


int adjoint_main(
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
    static double _del[N][N];

    void grid_function_calc(double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4],
        double v[N][N][4][2], double T[N][N], double *Kmax, double *Vmax,
        double PARAM_RAT, double* principal_angle_image, double* advection_velocity_image,
        double* diffusion_coefficient_image, double* correlation_time_image);
    void evolve_diffusion(double _del[N][N], double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4],
        double dt);
    void evolve_advection(double _del[N][N], double v[N][N][4][2], double dt);
    void evolve_decay(double _del[N][N], double T[N][N], double dt);
    void evolve_source(double del[N][N], double dt, double* source);

    double dx = PARAM_FOV/N;
    double dy = PARAM_FOV/N;

    /* correlation length l = sqrt(K*T) */
    /* correlation time is t = T */
    /* so diffusion coefficient is l^2/t */

    int i,j ;
    double Dtl = tf / (double) NUM_IMAGES;   /* image file output cadence */

    /* calculate some grid functions */
    static double v[N][N][4][2];
    double Kmax = 0.;
    double Vmax = 0.;
    static double T[N][N];
    double F_coeff_gradx[N][N][4] = {{{0.}}};
    double F_coeff_grady[N][N][4] = {{{0.}}};


    grid_function_calc(F_coeff_gradx, F_coeff_grady, v, T, &Kmax, &Vmax,
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
    gsl_rng * r;
    r = gsl_rng_alloc(gsl_rng_mt19937); /* Mersenne twister */
    gsl_rng_set(r, 0);
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
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
        evolve_source(_del, dt / Dtl, &source[n]);
        evolve_diffusion(_del, F_coeff_gradx, F_coeff_grady, dt);
        evolve_advection(_del, v, dt);
        evolve_decay(_del, T, dt);

        /* periodically execute diagnostics */
        if(t > tl) {
            /* check rms and output  random field */
            rms = 0.;
            for(i=0;i<N;i++)
            for(j=0;j<N;j++) {
                rms += _del[i][j]*_del[i][j] * Dtl;
                output_video[n] = _del[i][j];
                n++;
            }
            rms = sqrt(rms)/N;

            if (verbose) {
                fprintf(stderr,"%lf %lf\n",t, rms);
            }
            /* set time for next diagnostic output */
            tl += Dtl;
        }
        nstep++;
        t += dt;
    }
}



/* return the coordinates of a zone center */

void ij_to_xy(int i, int j, double *x, double *y)
{
    double dx = PARAM_FOV/N;
    double dy = PARAM_FOV/N;

    *x = (i - N/2)*dx ;
    *y = (j - N/2)*dy ;
}

/* return image coordinates */
void xy_image(double x[N][N], double y[N][N])
{
    int i,j;
    void ij_to_xy(int i, int j, double *x, double *y);

    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        ij_to_xy(i,j,&x[i][j],&y[i][j]);
    }

}

/* transform random field into fake image */
void apply_envelope(double _del[N][N], double fake_image[N][N], double* envelope_image, double PARAM_AMP)
{
    int i,j, k;
    for(i=0;i<N;i++) 
    for(j=0;j<N;j++) {
        fake_image[i][j] = envelope_image[k]*exp(-PARAM_AMP*_del[i][j]);
        k++;
    }
}


