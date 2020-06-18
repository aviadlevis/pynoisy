#include "param.h"

#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

const char model_name[] = "noisy_unif";

const double param_mass    = 1.; 
const double param_mdot    = 1.;
const double param_x0start = 0.;
const double param_x0end   = 1.; 
const double param_x1start = -0.5;
const double param_x1end   = 0.5;
const double param_x2start = -0.5;
const double param_x2end   = 0.5;

/* ratio of coefficients of x1 and x2 */
static const double param_rat = 0.2;

double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int ni, int nj, int nk,
		 int pi, int pj, int pk, double dx0, double dx1, double dx2)
{
  return raw;
}

void param_coeff(double* coeff, double x0, double x1, double x2, double dx0,
		  double dx1, double dx2, int index)
{
  double theta, gamma, beta, omega, t;
  theta = M_PI / 4.;

  gamma = 1.;                             /* correlation length */
  beta  = 1.;                             /* correlation time */
  
  t     = beta;
  gamma = gamma * gamma / beta;           /* diffusion coefficient */
  beta  = (1. - param_rat) * gamma;       /* real beta */
  gamma = param_rat * gamma;               /* real gamma */
  
  omega = 0.;
  
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
  int i, j, k;

  double* saved;
  saved = (double*) calloc(ni * npi * nj * npj, sizeof(double));
  for (i = 0; i < ni * npi * nj * npj; i ++)
    saved[i] = gsl_ran_gaussian_ziggurat(rstate, 1.);

  int l = 0;
  for (k = 0; k < nk; k++) {
    for (j = 0; j < nj; j++) {
      for (i = 0; i < ni; i++) {
	values[l] = saved[pj * nj * npi * ni + j * npi * ni + pi * ni + i];
	l++;
      }
    }
  }

  free(saved);
}
