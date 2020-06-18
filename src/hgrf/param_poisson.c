#include "param.h"

#include <math.h>

const char model_name[] = "poisson";

const double param_mass    = 1.;
const double param_mdot    = 1.;
const double param_x0start = 0.;
const double param_x0end   = 2. * M_PI;
const double param_x1start = 0.;
const double param_x1end   = 2. * M_PI;
const double param_x2start = 0.;
const double param_x2end   = 2. * M_PI;

double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int ni, int nj, int nk,
		 int pi, int pj, int pk, double dx0, double dx1, double dx2)
{
  return raw;
}

void param_coeff(double* coeff, double x0, double x1, double x2, double dx0,
		  double dx1, double dx2, int index)
{
  coeff[0] = 1. / (dx2 * dx2);
  coeff[1] = 1. / (dx1 * dx1);
  coeff[2] = 1. / (dx0 * dx0);
  coeff[3] = -2. * (coeff[0] + coeff[1] + coeff[2]);
}

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk,
		      int pi, int pj, int pk, int npi, int npj, int npk,
		      double dx0, double dx1, double dx2)
{
  int i;
  int nvalues = ni * nj * nk;

  double x0, x1, x2;

  int gridi, gridj, gridk;
  
  for (i = 0; i < nvalues; i++) {
    gridk = i / (ni * nj);
    gridj = (i - ni * nj * gridk) / ni;
    gridi = i - ni * nj * gridk + (pi - gridj) * ni;
    gridj += pj * nj;
    gridk += pk * nk;

    x0 = param_x0start + dx0 * gridk;
    x1 = param_x1start + dx1 * gridj;
    x2 = param_x2start + dx2 * gridi;
    
    values[i] = -3. * sin(x0) * sin(x1) * sin(x2);
  }
}
