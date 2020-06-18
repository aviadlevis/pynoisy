#include "param.h"

#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define SMALL 1.E-10

const char model_name[] = "disk_xy";

const double param_mass    = 1.E6;       /* in solar masses*/
const double param_mdot    = 1.E-7;      /* in solar masses per year */
const double param_x0start = 0.;         /* t in terms of M */
const double param_x0end   = 100.; 
const double param_x1start = -10.;       /* x in terms of M */
const double param_x1end   = 10.;
const double param_x2start = -10.;       /* y in terms of M */  
const double param_x2end   = 10.;

/* ratio of correlation length to local radius */
static const double param_lam = 3.;
/* product of correlation time and local Keplerian frequency */
static const double param_tau = 3.;
/* ratio of coefficients of x0 and phi */
static const double param_r02 = 0.01;
/* ratio of coefficients of r and phi */
static const double param_r12 = 0.1;
/* cutoff radius */
static const double param_rct = 1.;

double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int ni, int nj, int nk,
		 int pi, int pj, int pk, double dx0, double dx1, double dx2)
{
  double radius, x, y;

  x = param_x1start + (j + pj * nj) * dx1;
  y = param_x2start + (i + pi * ni) * dx2;
  
  radius = sqrt(x * x + y * y);
  if (radius >= param_rct)
    return 3. * param_mdot * 5.67E46 // solar mass*c^2/year to erg/s
      * ( 1. - sqrt( param_rct / radius ) )
      / (8. * M_PI * pow(radius, 3.) )
      * exp( 0.5 * (raw - avg_raw) / sqrt(var_raw) );
  else
    return 0.;
  /* in erg/s per unit area (in M^2) */
}

static double cutoff(double r, double r0, double fr0, double dfr0, double f0)
{
  double a, b;
  b = (2. * (fr0 - f0) - r0 * dfr0) / pow( r0, 3. );
  a = (fr0 - f0) / (b * r0 * r0) + r0;
  return b * r * r * (a - r) + f0;
}

static double w_keplerian(double x0, double x1, double x2)
{
  double r = sqrt(x1 * x1 + x2 * x2);

  if (r >= param_rct)
    return pow(r, -1.5);
  else
    return cutoff(r, param_rct, pow(param_rct, -1.5),
		  -1.5 * pow(param_rct, -2.5), 0.8 * pow(param_rct, -1.5));
}

static double corr_length(double x0, double x1, double x2)
{
  return param_lam;
  
  double r = sqrt(x1 * x1 + x2 * x2);
  
  if (r >= param_rct)
    return param_lam;
  else
    return cutoff(r, param_rct, param_lam, 0., param_lam * 0.8);
}

static double corr_time(double x0, double x1, double x2)
{
  double r = sqrt(x1 * x1 + x2 * x2);

  if (r >= param_rct)
    return 2. * M_PI * param_tau / fabs( w_keplerian(x0, x1, x2) );
  else
    return cutoff(r, param_rct,
		  2. * M_PI * param_tau * pow(param_rct, 1.5),
		  2. * M_PI * param_tau * 1.5 * sqrt(param_rct),
		  2. * M_PI * param_tau * 0.8 * pow(param_rct, 1.5) );
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
  double theta, psi, gamm0, gamm1, beta0, beta1, a, b, c, d, e;
  theta = -7. * M_PI / 18.;
  /* phi = arctan(omega) = arctan(sqrt(GM)/r^(3/2))
         = arctan(c^3/(GMe^(3x/2))) since r=GMe^x/c^2)
     c^3/(GM) is canceled out by unit conversion
  */
  psi = atan( w_keplerian(x0, x1, x2) );
  
  gamm0 = param_r02 * corr_time(x0, x1, x2);
  beta0 = (1. - param_r02) * corr_time(x0, x1, x2);
  gamm1 = param_r12 * corr_length(x0, x1, x2);
  beta1 = (1. - param_r12) * corr_length(x0, x1, x2);
  
  /* coefficients in log r coordinates */
  a = gamm0 + beta0 * sin(psi) * sin(psi)
        + gamm1 + beta1 * sin(theta) * sin(theta);
  b = 2. * beta1 * cos(theta) * sin(theta);
  c = gamm1 + beta1 * cos(theta) * cos(theta);
  d = 2. * beta0 * cos(psi) * sin(psi);
  e = gamm0 + beta0 * cos(psi) * cos(psi);

  /* dy^2 */
  coeff[0] = (c * x2 * x2 + b * x1 * x2 + a * x1 * x1) / (dx2 * dx2);
  /* dydx */
  coeff[1] = 0.5 * (c * x1 * x2 + 0.5 * b * (x1 * x1 - x2 * x2) - a * x1 * x2)
    / (dx1 * dx2);
  /* dx^2 */
  coeff[2] = (c * x1 * x1 - b * x1 * x2 + a * x2 * x2) / (dx1 * dx1);
  /* dydt */
  coeff[3] = 0.25 * d * x1 / (dx0 * dx2);
  /* dxdt */
  coeff[4] = -0.25 * d * x2 / (dx0 * dx1);
  /* dt^2 */
  coeff[5] = e / (dx0 * dx0);
  /* dy */
  coeff[6] = 0.5 * (c * x2 + b * x1 - a * x2) / dx2;
  /* dx */
  coeff[7] = 0.5 * (c * x1 - b * x2 - a * x1) / dx1;
  /* const */
  coeff[8] = -2. * ( coeff[0] + coeff[2] + coeff[5] ) - ksq(x0, x1, x2);
}

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk,
		      int pi, int pj, int pk, int npi, int npj, int npk,
		      double dx0, double dx1, double dx2)
{
  int i;
  int nvalues = ni * nj * nk;

  double x0, x1, x2, r;

  int gridi, gridj, gridk;

  for (i = 0; i < nvalues; i++) {
    gridk = i / (ni * nj);
    gridj = (i - ni * nj * gridk) / ni;
    gridi = i - ni * nj * gridk + (pi - gridj) * ni;
    gridj += pj * nj;
    gridk += pk * nk;

    x0 = param_x0start + dx0 + gridk;
    x1 = param_x1start + dx1 * gridj;
    x2 = param_x2start + dx2 * gridi;

    r = sqrt(x1 * x1 + x2 * x2);
    
    values[i] = pow(r, 1.5) * gsl_ran_gaussian_ziggurat(rstate, 1.);
  }
}
