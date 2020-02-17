
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

/* size of image.  notice that solution time scales as N^4 */
#define N  256

#define SMALL   1.e-10

/* set parameters of model */
#define PARAM_FOV   1.0     /* field of view */
#define PARAM_EPS   1.0     /* strength of forcing */
#define PARAM_AMP   0.05    /* strength of perturbation; image = exp(-AMP*_del)*envelope */
#define PARAM_LAM   0.5     /* ratio of correlation length to local radius */
#define PARAM_TAU   1.0     /* product of correlation time and local Keplerian frequency */
#define PARAM_RCH   0.2     /* scaling radius */
#define PARAM_RAT   0.1     /* ratio of diffusion coefficients */

