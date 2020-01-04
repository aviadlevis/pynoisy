
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

/*** this next bit describes the model ***/

double correlation_length(double x,double y)
{

    return( 1.0 );

}

double correlation_time(double x,double y)
{

    return( 1.0 );
}

double diffusion_coefficient(double x,double y)
{

    double K = 1.0;

    return( K );
}

/* return principal axes of diffusion tensor */

void principal_axis_func(double x, double y, double *e1x, double *e1y, double *e2x, double *e2y)
{
    double phi,s,c;

    phi = M_PI/4.;

    c = cos(phi);
    s = sin(phi);

    *e1x = c;
    *e1y = s;
    *e2x = -s;
    *e2y = c;
}

/* return advection velocity as a function of position */

void advection_velocity(double x, double y, double va[2])
{

    double vx = 0. ;
    double vy = 0. ;

    va[0] = vx ;
    va[1] = vy ; 
}


/* return envelope function; image is exp(-del)*envelope */

double envelope(double x, double y)
{
    return( 1.0 );
}

/* noise model */
void noise_model(double del_noise[][N], double dt)
{

    int i,j;
    static int first_call = 1;
    static gsl_rng * r;

#if 0
	/* white noise in time model */
    if(first_call) {
        r = gsl_rng_alloc(gsl_rng_mt19937); /* Mersenne twister */
        gsl_rng_set(r, 0);
        first_call = 0;
    }

    /* update del */
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        del[i][j] += PARAM_EPS*gsl_ran_gaussian_ziggurat(r,1.0);
    }
#endif

#if 1
	/* static model */
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
		del_noise[i][j] = dt*save_noise[i][j];
	}
#endif

}
