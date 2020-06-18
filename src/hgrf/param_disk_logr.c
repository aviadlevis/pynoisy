#include "param.h"

#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

const char model_name[] = "disk_logr";

const double param_mass    = 1.E6;       /* in solar masses*/
const double param_mdot    = 1.E-7;      /* in solar masses per year */
const double param_x0start = 0.;         /* t in terms of M */
const double param_x0end   = 100.; 
const double param_x1start = log(1.);    /* x1 = log(r), r in terms of M */
const double param_x1end   = log(10.);
const double param_x2start = 0.;
const double param_x2end   = 2. * M_PI;

/* log of ratio of correlation length to local radius */
static const double param_lam = 36.;
/* product of correlation time and local Keplerian frequency */
static const double param_tau = 150.;
/* ratio of coefficients of x0 and x2 */
static const double param_r02 = 0.01;
/* ratio of coefficients of x1 and x2 */
static const double param_r12 = 0.05;

double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int ni, int nj, int nk,
		 int pi, int pj, int pk, double dx0, double dx1, double dx2)
{
  double radius;
  radius = exp( param_x1start + (j + pj * nj) * dx1 );
  return 3. * param_mdot * 5.67E46 // solar mass per year to erg/s
    * ( 1. - sqrt( exp(param_x1start) / radius) )
    / ( 8. * M_PI * pow(radius , 3.) )
    * exp( 0.5 * (raw - avg_raw) / sqrt(var_raw) );

  /* envelope is 3/(8PI)*mdot*GM/r^3*(1-sqrt(r0/r))
     in units of erg/s per unit area (in M^2) */
}

static double w_keplerian(double x0, double x1, double x2)
{
  return exp(-1.5 * x1);
}

static double corr_length(double x0, double x1, double x2)
{
  return param_lam; /* a constant corr_length in x0 correponds to a linearly
		       increasing corr_length in r */
}

static double corr_time(double x0, double x1, double x2)
{
  return param_tau / w_keplerian(x0, x1, x2);
}

static double ksq(double x0, double x1, double x2)
{
  return 1.;
  //  return param_tau * param_r02 + param_r12 * corr_length(x0, x1, x2);
  // + log(1. + x1 * log(10.) / (2. * M_PI) );
}

void param_coeff(double* coeff, double x0, double x1, double x2, double dx0,
		  double dx1, double dx2, int index)
{
  double theta, psi, gamm0, gamm1, beta0, beta1;
  theta = -7. * M_PI / 18.;
  /* phi = arctan(omega) = arctan(sqrt(GM)/r^(3/2))
         = arctan(c^3/(GMe^(3x/2))) since r=GMe^x/c^2)
     c^3/(GM) is canceled out by unit conversion
  */
  psi   = atan( w_keplerian(x0, x1, x2) );

  gamm0 = param_r02 * corr_time(x0, x1, x2);
  beta0 = (1. - param_r12) * corr_time(x0, x1, x2);
  gamm1 = param_r12 * corr_length(x0, x1, x2);
  beta1 = (1. - param_r12) * corr_length(x0, x1, x2);
  
  /* dphi^2 */
  coeff[0] = ( gamm1 + beta1 * sin(theta) * sin(theta)
	       + gamm0 + beta0 * sin(psi) * sin(psi) ) / (dx2 * dx2);
  /* dphidx */
  coeff[1] = 0.5 * beta1 * cos(theta) * sin(theta) / (dx1 * dx2);
  /* dx^2 */
  coeff[2] = ( gamm1 + beta1 * cos(theta) * cos(theta) ) / (dx1 * dx1);
  /* dphidt */
  coeff[3] = 0.5 * beta0 * cos(psi) * sin(psi) / (dx2 * dx0);
  /* dt^2 */
  coeff[4] = ( gamm0 + beta0 * cos(psi) * cos(psi) ) / (dx0 * dx0);
  /* const */
  coeff[5] = -2. * ( coeff[0] + coeff[2] + coeff[4] ) - ksq(x0, x1, x2);
}

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk,
		      int pi, int pj, int pk, int npi, int npj, int npk,
		      double dx0, double dx1, double dx2)
{
  int i;
  int nvalues = ni * nj * nk;

  for (i = 0; i < nvalues; i++) {
    values[i] = gsl_ran_gaussian_ziggurat(rstate, 1.);
  }
}
