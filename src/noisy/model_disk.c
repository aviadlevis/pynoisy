
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

/* correlation length is proportional to local radius */
double correlation_length(double x,double y,double PARAM_RCH,double PARAM_LAM)
{
    double r = sqrt( (x*x + y*y)/(PARAM_RCH*PARAM_RCH) );

    return( PARAM_LAM * r );

}

void get_correlation_length_image(int nx, int ny, double* correlation_length_image, double PARAM_RCH, double PARAM_LAM)
{
    int i,j;
    double x,y;
    void ij_to_xy(int nx, int ny, int i, int j, double *x, double *y);
    double correlation_length(double x,double y, double PARAM_RCH, double PARAM_LAM);

    int k = 0;
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        ij_to_xy(nx,ny,i,j,&x,&y);
        correlation_length_image[k] = correlation_length(x, y, PARAM_RCH, PARAM_LAM);
        k++;
    }
}

/* correlation time is proportional to the local Keplerian time */
double correlation_time(double x,double y, double PARAM_TAU, double PARAM_RCH)
{
    double W_Keplerian(double x, double y, double direction, double PARAM_RCH);

    double t = 1./W_Keplerian(x,y, -1.0, PARAM_RCH);

    return( PARAM_TAU * fabs(t) );
}

void get_correlation_time_image(int nx, int ny, double* correlation_time_image, double PARAM_TAU, double PARAM_RCH)
{
    int i,j;
    double x,y;
    void ij_to_xy(int nx, int ny, int i, int j, double *x, double *y);
    double correlation_time(double x,double y, double PARAM_TAU, double PARAM_RCH);

    int k = 0;
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        ij_to_xy(nx,ny,i,j,&x,&y);
        correlation_time_image[k] = correlation_time(x, y, PARAM_TAU, PARAM_RCH);
        k++;
    }
}


double diffusion_coefficient(double x, double y, double PARAM_TAU, double PARAM_RCH, double PARAM_LAM)
{
    double l = correlation_length(x, y, PARAM_RCH, PARAM_LAM);
    double t = correlation_time(x, y, PARAM_TAU, PARAM_RCH);
    double K = l*l/t;

    return( 2.*K );
}

void get_diffusion_coefficient(int nx, int ny, double* diffusion_coefficient_image, double PARAM_TAU, double PARAM_LAM, double PARAM_RCH)
{
    int i,j;
    double x,y;
    void ij_to_xy(int nx, int ny, int i, int j, double *x, double *y);
    double diffusion_coefficient(double x, double y, double PARAM_TAU, double PARAM_RCH, double PARAM_LAM);

    int k = 0;
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        ij_to_xy(nx,ny,i,j,&x,&y);
        diffusion_coefficient_image[k] = diffusion_coefficient(x, y, PARAM_TAU, PARAM_RCH, PARAM_LAM);
        k++;
    }
}

/* return principal axes of diffusion tensor */

void principal_axis_func(double x, double y, double *e1x, double *e1y, double *e2x, double *e2y, double opening_angle)
{
    double phi,s,c;
    double phi_func(double x, double y, double opening_angle);

    phi = phi_func(x,y, opening_angle);

    c = cos(phi);
    s = sin(phi);

    *e1x = c;
    *e1y = s;
    *e2x = -s;
    *e2y = c;
}

/* return Keplerian orbital frequency */
double W_Keplerian(double x, double y, double direction, double PARAM_RCH)
{

    /* negative W0 turn clockwise on image */
    /* 2.*M_PI allows W to be expressed in terms of orbital period */
    double period = 1. ;
    double W0 = direction * 2.*M_PI / period ;

    /* Keplerian velocity field */
    double r = sqrt(x*x + y*y)/PARAM_RCH + 0.5;
    double W = W0*pow(r, -1.5);

    return(W);
}


/* return advection velocity as a function of position */

void advection_velocity(double x, double y, double* va, double direction, double PARAM_RCH)
{
    double r,W,q,taper,rmax;
    double W_Keplerian(double x, double y, double direction, double PARAM_RCH);
    double W_Keplerian(double x, double y, double direction, double PARAM_RCH);

    W = W_Keplerian(x, y, direction, PARAM_RCH);

    /* taper velocity with bump function */
    r = sqrt(x*x + y*y);
    rmax = 1.0/2.;  /* radius where velocity field is turned off */
    q = 1. - r*r/(rmax*rmax);
    taper = (r >= rmax) ? 0. : exp(-1./q) ;
    W = W*taper;

    va[0] = -W*y ;
    va[1] = W*x ;
}

/* return advection velocity for the whole image */
void get_advection_velocity_image(int nx, int ny, double* velocity, double direction, double PARAM_RCH)
{
    int i,j;
    double x,y;
    void ij_to_xy(int nx, int ny, int i, int j, double *x, double *y);
    void advection_velocity(double x, double y, double va[2], double direction, double PARAM_RCH);
    int k = 0;
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        ij_to_xy(nx,ny,i,j,&x,&y);
        advection_velocity(x, y, &velocity[k], direction, PARAM_RCH);
        k = k+2;
    }
}

/* return angle of principle axes of diffusion tensor
   with respect to local radius vector.

   this can be used to adjust the opening angle of spirals */

double phi_func(double x, double y, double opening_angle)
{
    double phi0 = atan2(y, x);
    double phi = phi0 + opening_angle ;
    return(phi);
}

/* return diffusion tensor principal angle for the whole image */
void principal_angle_image(int nx, int ny, double* angle, double opening_angle)
{
    int i,j;
    double x,y;
    void ij_to_xy(int nx, int ny, int i, int j, double *x, double *y);
    double phi_func(double x, double y, double opening_angle);

    int k=0;
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        ij_to_xy(nx,ny,i,j,&x,&y);
        angle[k] = phi_func(x, y, opening_angle);
        k++;
    }
}


/* return envelope function; image is exp(-_del)*envelope */

double envelope(double x, double y, double PARAM_RCH)
{
    double r = sqrt(x*x + y*y)/PARAM_RCH + SMALL;
    double ir = 1./r ;

    double f1 = exp(-ir*ir);      /* suppresses image inside PARAM_RCH */
    double f2 = pow(ir, 4.) ;     /* outer power-law envelope */

    return(f1*f2);
}

/* return image envelope */
void get_envelope_image(int nx, int ny, double* envelope_image, double PARAM_RCH)
{
    int i,j;
    double x,y;
    void ij_to_xy(int nx, int ny, int i, int j, double *x, double *y);
    double envelope(double x, double y, double PARAM_RCH);

    int k=0;
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        ij_to_xy(nx, ny, i, j, &x, &y);
        envelope_image[k] = envelope(x, y, PARAM_RCH);
        k++;
    }
}

/* noise model */
void noise_model(int nx, int ny,double del_noise[nx][ny], double dt, double PARAM_EPS, gsl_rng* r)
{
    int i,j;

    /* white noise in space and time model */
#if 1
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        del_noise[i][j] = PARAM_EPS*gsl_ran_gaussian_ziggurat(r, 1.0);
    }
#endif

#if 0
    /* white noise in space, static in time model */
    static double save_noise[nx][ny];
    if(first_call) {
        r = gsl_rng_alloc(gsl_rng_mt19937); /* Mersenne twister */
        gsl_rng_set(r, 0);

        for(i=0;i<nx;i++)
        for(j=0;j<ny;j++) {
            save_noise[i][j] = PARAM_EPS*gsl_ran_gaussian_ziggurat(r,1.0);
        }

        first_call = 0;
    }
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        del_noise[i][j] = save_noise[i][j];
    }
#endif

}

