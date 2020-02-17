
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
double correlation_length(double x,double y)
{
    double r = sqrt( (x*x + y*y)/(PARAM_RCH*PARAM_RCH) );

    return( PARAM_LAM * r );

}

/* correlation time is proportional to the local Keplerian time */
double correlation_time(double x,double y)
{
    double W_Keplerian(double x, double y);

    double t = 1./W_Keplerian(x,y);

    return( PARAM_TAU * fabs(t) );
}

double diffusion_coefficient(double x,double y)
{

    double l = correlation_length(x, y);
    double t = correlation_time(x, y);
    double K = l*l/t;

    return( 2.*K );
}

/* return principal axes of diffusion tensor */

void principal_axis_func(double x, double y, double *e1x, double *e1y, double *e2x, double *e2y)
{
    double phi,s,c;
    double phi_func(double x, double y);

    phi = phi_func(x,y);

    c = cos(phi);
    s = sin(phi);

    *e1x = c;
    *e1y = s;
    *e2x = -s;
    *e2y = c;
}

/* return Keplerian orbital frequency */
double W_Keplerian(double x, double y)
{

    /* negative W0 turn clockwise on image */
    /* 2.*M_PI allows W to be expressed in terms of orbital period */
    double direction = -1. ;
    double period = 1. ;
    double W0 = direction * 2.*M_PI / period ;

    /* Keplerian velocity field */
    double r = sqrt(x*x + y*y)/PARAM_RCH + 0.5;
    double W = W0*pow(r, -1.5);

    return(W);
}


/* return advection velocity as a function of position */

void advection_velocity(double x, double y, double va[2])
{
    double r,W,q,taper,rmax;
    double W_Keplerian(double , double y);

    W = W_Keplerian(x,y);

    /* taper velocity with bump function */
    r = sqrt(x*x + y*y);
    rmax = PARAM_FOV/2.;  /* radius where velocity field is turned off */
    q = 1. - r*r/(rmax*rmax);
    taper = (r >= rmax) ? 0. : exp(-1./q) ;
    W = W*taper;

    va[0] = -W*y ;
    va[1] = W*x ;
}

/* return advection velocity for the whole image */

void advection_velocity_image(double velocity[N][N][2])
{
    int i,j;
    double x,y;
    void ij_to_xy(int i, int j, double *x, double *y);
    void advection_velocity(double x, double y, double va[2]);

    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        ij_to_xy(i,j,&x,&y);
        advection_velocity(x, y, velocity[i][j]);
    }
}


/* return angle of principle axes of diffusion tensor
   with respect to local radius vector.

   this can be used to adjust the opening angle of spirals */

double phi_func(double x, double y)
{
    double phi0 = atan2(y, x);
    double opening_angle = 0.3*M_PI;
    double phi = phi0 + opening_angle ;

    return(phi);
}


/* return envelope function; image is exp(-_del)*envelope */

double envelope(double x, double y)
{
    double r = sqrt(x*x + y*y)/PARAM_RCH + SMALL;
    double ir = 1./r ;

    double f1 = exp(-ir*ir);      /* suppresses image inside PARAM_RCH */
    double f2 = pow(ir, 4.) ;     /* outer power-law envelope */

    return(f1*f2);
}

/* noise model */
void noise_model(double del_noise[N][N], double dt)
{

    int i,j;
    static int first_call = 1;
    static gsl_rng * r;

    /* white noise in space and time model */
#if 1
    if(first_call) {
        r = gsl_rng_alloc(gsl_rng_mt19937); /* Mersenne twister */
        gsl_rng_set(r, 0);
        first_call = 0;
    }

    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        del_noise[i][j] = PARAM_EPS*gsl_ran_gaussian_ziggurat(r,1.0);
    }
#endif

#if 0
    /* white noise in space, static in time model */
    static double save_noise[N][N];
    if(first_call) {
        r = gsl_rng_alloc(gsl_rng_mt19937); /* Mersenne twister */
        gsl_rng_set(r, 0);

        for(i=0;i<N;i++)
        for(j=0;j<N;j++) {
            save_noise[i][j] = PARAM_EPS*gsl_ran_gaussian_ziggurat(r,1.0);
        }

        first_call = 0;
    }
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        del_noise[i][j] = save_noise[i][j];
    }
#endif

}

