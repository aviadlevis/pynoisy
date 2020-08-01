#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define SMALL 1.E-10

const char model_name[] = "general_xy";

const double param_mass    = 1.E6;       /* in solar masses*/
const double param_mdot    = 1.E-7;      /* in solar masses per year */
const double param_x0start = 0.;         /* t in terms of M */
const double param_x0end   = 100.;
const double param_x1start = -10.;       /* x in terms of M */
const double param_x1end   = 10.;
const double param_x2start = -10.;       /* y in terms of M */
const double param_x2end   = 10.;

/*
ratio of correlation length to local radius
static const double param_lam = 3.;

product of correlation time and local Keplerian frequency
static const double param_tau = 3.;

cutoff radius
static const double param_rct = 1.;

ratio of coefficients of temporal vs spatial correlation
static const double param_r02 = 0.01;

ratio of coefficients of major and minor axes of spatial correlation
static const double param_r12 = 0.1;
*/

/* smooth cutoff at radius r0, where function has value f(r0) and slope
df(r0). continuous + once differentiable at r0, and has value f(0) and
slope 0 at r = 0 */

static double cutoff(double r, double r0, double fr0, double dfr0, double f0)
{
  double a, b;
  b = (2. * (fr0 - f0) - r0 * dfr0) / pow( r0, 3. );
  a = (fr0 - f0) / (b * r0 * r0) + r0;
  return b * r * r * (a - r) + f0;
}


static double w_keplerian(double x0, double x1, double x2, double param_rct)
{
  double r = sqrt(x1 * x1 + x2 * x2);

  if (r >= param_rct)
    return pow(r, -1.5);
  else
    return cutoff(r, param_rct, pow(param_rct, -1.5),
		  -1.5 * pow(param_rct, -2.5), 0.9 * pow(param_rct, -1.5));
}


/* unit vector in direction of spatial correlation */
static double get_spatial_angle(double x0, double x1, double x2, double param_rct, double param_theta)
{
  return atan2(x2, x1) + param_theta;
}

static void get_velocity(double x0, double x1, double x2, double* v, double direction, double param_rct)
{
  double omega = direction * w_keplerian(x0, x1, x2, param_rct);
  v[0] = -x2 * omega;
  v[1] = x1 * omega;
}

double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int ni, int nj, int nk,
		 int pi, int pj, int pk, double dx0, double dx1, double dx2, double param_rct)
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


static double corr_length(double x0, double x1, double x2, double param_rct, double param_lam)
{
  /* return param_lam; */

  double r = sqrt(x1 * x1 + x2 * x2);

  if (r >= param_rct)
    return param_lam * r;
  else
    return cutoff(r, param_rct, param_lam * param_rct,
		  param_lam, 0.9 * param_lam * param_rct);
}

static double corr_time(double x0, double x1, double x2, double param_tau, double param_rct)
{
  /* return param_tau; */

  double r = sqrt(x1 * x1 + x2 * x2);

  if (r >= param_rct)
    return 2. * M_PI * param_tau / fabs( w_keplerian(x0, x1, x2, param_rct) );
  else
    return cutoff(r, param_rct,
		  2. * M_PI * param_tau * pow(param_rct, 1.5),
		  2. * M_PI * param_tau * 1.5 * sqrt(param_rct),
		  0.9 * 2. * M_PI * param_tau * pow(param_rct, 1.5) );
}


void get_correlation_length_image(int ni, int nj, double* correlation_length_image, double param_rct, double param_lam)
{
  int i, j;
  double x0, x1, x2;
  double dx0, dx1, dx2;

  model_set_spacing(&dx0, &dx1, &dx2, ni, nj, 1, 1, 1, 1);
  int k = 0;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
        x0 = param_x0start + dx0 + 0;
        x1 = param_x1start + dx1 * i;
        x2 = param_x2start + dx2 * j;
        correlation_length_image[k] = corr_length(x0, x1, x2, param_rct, param_lam);
        k++;
    }
  }
}


void get_correlation_time_image(int ni, int nj, double* correlation_time_image, double param_tau, double param_rct)
{
  int i, j;
  double x0, x1, x2;
  double dx0, dx1, dx2;

  model_set_spacing(&dx0, &dx1, &dx2, ni, nj, 1, 1, 1, 1);
  int k = 0;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
        x0 = param_x0start + dx0 + 0;
        x1 = param_x1start + dx1 * i;
        x2 = param_x2start + dx2 * j;
        correlation_time_image[k] = corr_time(x0, x1, x2, param_tau, param_rct);
        k++;
    }
  }
}

void get_spatial_angle_image(int ni, int nj, double* spatial_angle_image, double param_rct, double param_theta)
{
  int i, j;
  double x0, x1, x2;
  double dx0, dx1, dx2;

  model_set_spacing(&dx0, &dx1, &dx2, ni, nj, 1, 1, 1, 1);
  int k = 0;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
        x0 = param_x0start + dx0 + 0;
        x1 = param_x1start + dx1 * i;
        x2 = param_x2start + dx2 * j;
        spatial_angle_image[k] = get_spatial_angle(x0, x1, x2, param_rct, param_theta);
        k++;
    }
  }
}

/* return velocity for the whole image */
void get_velocity_image(int ni, int nj, double* velocity, double direction, double param_rct)
{
      int i, j;
      double x0, x1, x2;
      double dx0, dx1, dx2;
      void get_velocity(double x0, double x1, double x2, double va[2], double direction, double param_rct);
      model_set_spacing(&dx0, &dx1, &dx2, ni, nj, 1, 1, 1, 1);
      int k = 0;
      for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            x0 = param_x0start + dx0 + 0;
            x1 = param_x1start + dx1 * i;
            x2 = param_x2start + dx2 * j;
            get_velocity(x0, x1, x2, &velocity[k], direction, param_rct);
            k=k+2;
        }
      }
}

void model_set_spacing(double* dx0, double* dx1, double* dx2,
		       int ni, int nj, int nk, int npi, int npj, int npk)
{
  *dx0 = (param_x0end - param_x0start) / (npk * nk);
  *dx1 = (param_x1end - param_x1start) / (npj * nj);
  *dx2 = (param_x2end - param_x2start) / (npi * ni);
}

