#include "param.h"

#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define SMALL 1.e-10

const char model_name[] = "noisy_disk";

const double param_mass    = 1.; 
const double param_mdot    = 1.;
const double param_x0start = 0.;
const double param_x0end   = 1.; 
const double param_x1start = -0.5;
const double param_x1end   = 0.5;
const double param_x2start = -0.5;
const double param_x2end   = 0.5;

/* strength of perturbation */
static const double param_amp = 0.05;
/* ratio of correlation length to local radius */
static const double param_lam = 0.01;
/* product of correlation time and local Keplerian frequency */
static const double param_tau = 0.01;
/* scaling radius */
static const double param_rch = 0.2;
/* ratio of diffusion coefficients */
static const double param_rat = 0.2;

double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int ni, int nj, int nk,
		 int pi, int pj, int pk, double dx0, double dx1, double dx2)
{
  double ir, x, y;
  
  x = param_x1start + (j + pj * nj) * dx1;
  y = param_x2start + (i + pi * ni) * dx2;
  
  ir = sqrt(x * x + y * y) / param_rch + SMALL;
  ir = fmin(param_x1end - param_x1start, param_x2end - param_x2start)
    * 0.5 / ir;

  return exp(-ir * ir) * pow(ir, 4.)
    * exp( -1. * param_amp * (raw - avg_raw) / sqrt(var_raw) );
}

static double w_keplerian(double x0, double x1, double x2)
{
  double r = sqrt(x1 * x1 + x2 * x2);
  
  return -4. * M_PI * pow(r / param_rch + 0.5, -1.5); 
}

static double corr_length(double x0, double x1, double x2)
{
  double r = sqrt(x1 * x1 + x2 * x2);

  return param_lam * r / (param_rch * param_rch);
}

static double corr_time(double x0, double x1, double x2)
{
  return param_tau * fabs( 1. / w_keplerian(x0, x1, x2) );
}

void param_coeff(double* coeff, double x0, double x1, double x2, double dx0,
		  double dx1, double dx2, int index)
{
  double theta, gamma, beta, omega, t, r, rmax, q;
  theta = atan2(x2, x1) + 0.2 * M_PI;

  gamma = corr_length(x0, x1, x2);        /* correlation length */
  beta  = corr_time(x0, x1, x2);          /* correlation time */
  omega = w_keplerian(x0, x1, x2);
  
  t     = beta + SMALL;
  gamma = gamma * gamma / beta;           /* diffusion coefficient */
  beta  = (1. - param_rat) * gamma;       /* real beta */
  gamma = param_rat * gamma;              /* real gamma */
  
  rmax  = fmin(param_x1end - param_x1start, param_x2end - param_x2start) / 2.;
  r     = sqrt(x1 * x1 + x2 * x2);
  q     = 1. - r * r / (rmax * rmax);
  omega = omega * ( (r >= rmax) ? 0. : exp(-1. / q) );
  
  coeff[0] = ( gamma + beta * sin(theta) * sin(theta) ) / (dx2 * dx2);
  coeff[1] = 0.5 * beta * cos(theta) * sin(theta) / (dx1 * dx2);
  coeff[2] = ( gamma + beta * cos(theta) * cos(theta) ) / (dx1 * dx1);
  coeff[3] = -0.5 * -omega * x2 / dx1;
  coeff[4] = -0.5 * omega * x1 / dx2;
  coeff[5] = -0.5 / dx0;
  coeff[6] = -2. * ( coeff[0] + coeff[2] ) - 1. / t;
}

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk,
		      int pi, int pj, int pk, int npi, int npj, int npk,
		      double dx0, double dx1, double dx2)
{
  int i;
  int nvalues = ni * nj * nk;
  
  for (i = 0; i < nvalues; i++)
    values[i] = gsl_ran_gaussian_ziggurat(rstate, 1.);
}
