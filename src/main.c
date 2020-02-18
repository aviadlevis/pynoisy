
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
    double opening_angle,
    double direction,
    double PARAM_RCH,
    double PARAM_FOV,
    double PARAM_LAM,
    double PARAM_TAU,
    double PARAM_RAT,
    double PARAM_AMP,
    double PARAM_EPS
    )
{
    static double _del[N][N];
    static double fake_image[N][N];
    void grid_function_calc(double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4], 
        double v[N][N][4][2], double T[N][N], double *Kmax, double *Vmax,  double opening_angle, double direction, double PARAM_RCH, double PARAM_FOV, double PARAM_LAM, double PARAM_TAU, double PARAM_RAT);
    void evolve_diffusion(double _del[N][N], double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4],
        double dt, double PARAM_FOV);
    void evolve_advection(double _del[N][N], double v[N][N][4][2], double dt, double PARAM_FOV);
    void evolve_decay(double _del[N][N], double T[N][N], double dt);
    void evolve_noise(double _del[N][N], double dt, double PARAM_EPS);

    double dx = PARAM_FOV/N;
    double dy = PARAM_FOV/N;

    double tf = 0.1;   /* length of evolution */

    /* correlation length l = sqrt(K*T) */
    /* correlation time is t = T */
    /* so diffusion coefficient is l^2/t */

    int i,j ;
    double Dtl = tf/400.;   /* image file output cadence */
    void apply_envelope(double _del[N][N], double fake_image[N][N], double PARAM_RCH, double PARAM_AMP, double PARAM_FOV);
    void emit_image(double fake_image[N][N], int n);
    void ij_to_xy(int i, int j, double *x, double *y, double PARAM_FOV);

    /* calculate some grid functions */
    static double v[N][N][4][2];
    double Kmax = 0.;
    double Vmax = 0.;
    static double T[N][N];
    double F_coeff_gradx[N][N][4] = {{{0.}}};
    double F_coeff_grady[N][N][4] = {{{0.}}};
    grid_function_calc(F_coeff_gradx, F_coeff_grady, v, T, &Kmax, &Vmax, opening_angle, direction, PARAM_RCH, PARAM_FOV, PARAM_LAM, PARAM_TAU, PARAM_RAT);
    fprintf(stderr,"Vmax: %g\n",Vmax);

    /* now that we know Kmax and Vmax, set timestep */
    double d = fmin(dx,dy);
        /* courant-limited timestep for diffusive part.  cour = 1 is
           the rigorous stability limit if dx = dy, RAT = 1 */
    double cour = 0.45;
    double dtdiff = cour*0.25*d*d/Kmax;   
    double dtadv = cour*0.5*d/Vmax;
    double dt = fmin(dtdiff, dtadv);
    fprintf(stderr,"dt,dtdiff,dtadv: %g %g %g\n",dt, dtdiff, dtadv);

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
    double tl = 0.;
    double F;
    while(t < tf){

        /* periodically execute diagnostics */
        if(t > tl) {
            /* check rms of random field */
            rms = 0.;
            for(i=0;i<N;i++)
            for(j=0;j<N;j++) rms += _del[i][j]*_del[i][j];
            rms = sqrt(rms)/N;

            /* transform random field into image */
            apply_envelope(_del, fake_image, PARAM_RCH, PARAM_AMP, PARAM_FOV);

            /* get light curve */
            F = 0.;
            for(i=0;i<N;i++)
            for(j=0;j<N;j++) F += fake_image[i][j]*dx*dy;
            fprintf(stderr,"%lf %lf %lf\n",t, F, rms);

            /* output image */
            emit_image(fake_image, n);

            /* set time for next diagnostic output */
            tl += Dtl;
            n++;
        }

        /* operator split */
        evolve_noise(_del, dt, PARAM_EPS);
        evolve_diffusion(_del, F_coeff_gradx, F_coeff_grady, dt, PARAM_FOV);
        evolve_advection(_del, v, dt, PARAM_FOV);
        evolve_decay(_del, T, dt);

        nstep++;
        t += dt;

    }

    FILE *fp = fopen("noisy.out", "w");
    if(fp == NULL) exit(1);
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) 
        fprintf(fp,"%d %d %lf\n",i,j,_del[i][j]);

}


/* return the coordinates of a zone center */

void ij_to_xy(int i, int j, double *x, double *y, double PARAM_FOV)
{
    double dx = PARAM_FOV/N;
    double dy = PARAM_FOV/N;

    *x = (i - N/2)*dx ;
    *y = (j - N/2)*dy ;
}

/* return image coordinates */
void xy_image(double x[N][N], double y[N][N], double PARAM_FOV)
{
    int i,j;
    void ij_to_xy(int i, int j, double *x, double *y, double PARAM_FOV);

    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        ij_to_xy(i,j,&x[i][j],&y[i][j], PARAM_FOV);
    }

}

/* transform random field into fake image */

void apply_envelope(double _del[N][N], double fake_image[N][N], double PARAM_RCH, double PARAM_AMP, double PARAM_FOV)
{

    int i,j;
    double x,y;
    double envelope(double x, double y, double PARAM_RCH);
    void ij_to_xy(int i, int j, double *x, double *y, double PARAM_FOV);

    for(i=0;i<N;i++) 
    for(j=0;j<N;j++) {
        ij_to_xy(i,j,&x,&y, PARAM_FOV);
        fake_image[i][j] = envelope(x,y,PARAM_RCH)*exp(-PARAM_AMP*_del[i][j]);
    }
}


