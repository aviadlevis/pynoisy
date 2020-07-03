#include "param.h"

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

int min(int x, int y)
{
  return (x < y) ? x : y;
}

int max(int x, int y)
{
  return (x > y) ? x : y;
}

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
  return atan2(x2, x1) + copysign( param_theta, w_keplerian(x0, x1, x2, param_rct) );
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


/* time correlation vector (1, v1, v2) */
static void set_u0(double* u0, double x0, double x1, double x2, double* v)
{
  u0[0] = 1.;
  u0[1] = v[1];
  u0[2] = v[0];
}

/* unit vectors in direction of major and minor axes */
static void set_u1_u2(double* u1, double* u2, double x0, double x1, double x2, double theta)
{

  u1[0] = 0.;
  u2[0] = 0.;

  if (x1 == 0 && x2 == 0) {
    u1[1] = 0.;
    u1[2] = 0.;

    u2[1] = 0.;
    u2[2] = 0.;
  }
  else {

    u1[1] = cos(theta);
    u1[2] = sin(theta);

    u2[1] = -sin(theta);
    u2[2] = cos(theta);
  }
}

static void set_h(double h[3][3], double x0, double x1, double x2, double param_r12,
                  double spatial_angle, double* velocity, double correlation_time, double correlation_length)
{
  int i, j;
  double u0[3], u1[3], u2[3];

  set_u0(u0, x0, x1, x2, velocity);
  set_u1_u2(u1, u2, x0, x1, x2, spatial_angle);
  
  double lam0, lam1, lam2; /* temporal, major, minor correlation lengths */

  lam0 = correlation_time;
  lam1 = correlation_length;
  lam2 = param_r12 * lam1;

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      h[i][j] = lam0 * lam0 * u0[i] * u0[j]
	            + lam1 * lam1 * u1[i] * u1[j]
	            + lam2 * lam2 * u2[i] * u2[j];
    }
  }
}


/* dh[0][2][1] = dh[0][2]/dx[1] */
static void set_dh(double dh[][3][3], double x0, double x1, double x2,
		   double dx0, double dx1, double dx2, double param_r12, int ni, int nj, double spatial_angle[ni][nj],
		   double velocity[ni][nj][2], double correlation_time[ni][nj], double correlation_length[ni][nj], int gridi, int gridj)
{
  int i, j, k;

  double dx[3] = {dx0, dx1, dx2};
  
  /* hm[0][2][1] = h(x0 - dx0, x1, x2)[2][1] */
  double hm[3][3][3], hp[3][3][3];
  set_h(hm[0], x0 - dx0, x1, x2, param_r12, spatial_angle[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)], velocity[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)], correlation_time[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)], correlation_length[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)]);
  set_h(hp[0], x0 + dx0, x1, x2, param_r12, spatial_angle[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)], velocity[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)], correlation_time[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)], correlation_length[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)]);
  set_h(hm[1], x0, x1 - dx1, x2, param_r12, spatial_angle[min(max(gridi,0), ni-1)][min(max(gridj-1,0), nj-1)], velocity[min(max(gridi,0), ni-1)][min(max(gridj-1,0), nj-1)], correlation_time[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)], correlation_length[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)]);
  set_h(hp[1], x0, x1 + dx1, x2, param_r12, spatial_angle[min(max(gridi,0), ni-1)][min(max(gridj+1,0), nj-1)], velocity[min(max(gridi,0), ni-1)][min(max(gridj+1,0), nj-1)], correlation_time[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)], correlation_length[min(max(gridi,0), ni-1)][min(max(gridj,0), nj-1)]);
  set_h(hm[2], x0, x1, x2 - dx2, param_r12, spatial_angle[min(max(gridi-1,0), ni-1)][min(max(gridj,0), nj-1)], velocity[min(max(gridi-1,0), ni-1)][min(max(gridj,0), nj-1)], correlation_time[min(max(gridi-1,0), ni-1)][min(max(gridj,0), nj-1)], correlation_length[min(max(gridi-1,0), ni-1)][min(max(gridj,0), nj-1)]);
  set_h(hp[2], x0, x1, x2 + dx2, param_r12, spatial_angle[min(max(gridi+1,0), ni-1)][min(max(gridj,0), nj-1)], velocity[min(max(gridi+1,0), ni-1)][min(max(gridj,0), nj-1)], correlation_time[min(max(gridi+1,0), ni-1)][min(max(gridj,0), nj-1)], correlation_length[min(max(gridi+1,0), ni-1)][min(max(gridj,0), nj-1)]);

  for (i = 0; i < 3; i++)
    for (j = 0; j < 3; j++)
      for (k = 0; k < 3; k++)
	dh[i][j][k] = 0.5 * ( hp[k][i][j] - hm[k][i][j] ) / dx[k];
}

static double ksq(double x0, double x1, double x2)
{
  return 1.;
  //  return param_tau * param_r02 + param_r12 * corr_length(x0, x1, x2);
  // + log(1. + x1 * log(10.) / (2. * M_PI) );
}

void param_coeff(double* coeff, double x0, double x1, double x2, double dx0, double dx1, double dx2,
                 double param_r12, int ni, int nj, double spatial_angle[ni][nj], double velocity[ni][nj][2],
                 double correlation_time[ni][nj], double correlation_length[ni][nj], int gridi, int gridj)
{
  double h[3][3], dh[3][3][3];
  set_h(h, x0, x1, x2, param_r12, spatial_angle[gridi][gridj], velocity[gridi][gridj], correlation_time[gridi][gridj], correlation_length[gridi][gridj]);
  set_dh(dh, x0, x1, x2, dx0, dx1, dx2, param_r12, ni, nj, spatial_angle, velocity, correlation_time, correlation_length, gridi, gridj);

  /* dy^2 */
  coeff[0] = -h[2][2] / (dx2 * dx2);
  /* dydx */
  coeff[1] = -0.5 * h[1][2] / (dx1 * dx2);
  /* dx^2 */
  coeff[2] = -h[1][1] / (dx1 * dx1);
  /* dydt */
  coeff[3] = -0.5 * h[0][2] / (dx0 * dx2);
  /* dxdt */
  coeff[4] = -0.5 * h[0][1] / (dx0 * dx1);
  /* dt^2 */
  coeff[5] = -h[0][0] / (dx0 * dx0);
  /* dy */
  coeff[6] = -0.5 * ( dh[0][2][0] + dh[1][2][1] + dh[2][2][2] ) / dx2;
  /* dx */
  coeff[7] = -0.5 * ( dh[0][1][0] + dh[1][1][1] + dh[2][1][2] ) / dx1;
  /* dt */
  coeff[8] = -0.5 * ( dh[0][0][0] + dh[1][0][1] + dh[2][0][2] ) / dx0;
  /* const */
  coeff[9] = -2. * ( coeff[0] + coeff[2] + coeff[5] ) + ksq(x0, x1, x2);
}


void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk,
		      int pi, int pj, int pk, int npi, int npj, int npk,
		      double dx0, double dx1, double dx2)
{
  int i;
  int nvalues = ni * nj * nk;

  double x0, x1, x2, scaling;

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

    //    double r = sqrt(x1 * x1 + x2 * x2);

    /* white noise scaled by 4th root of determinant of Lambda
       det Lambda = l0^2*l1^2*l2^2 */
    scaling = corr_time(x0, x1, x2, 1.0, 0.5) * corr_length(x0, x1, x2, 0.5, 5.0)
      * 0.1 * corr_length(x0, x1, x2, 0.5, 5.0);
    scaling = sqrt(scaling);

    values[i] = gsl_ran_gaussian_ziggurat(rstate, 1.) * scaling;
  }
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

