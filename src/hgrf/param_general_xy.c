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

/* ratio of correlation length to local radius */
static const double param_lam = 3.;
/* product of correlation time and local Keplerian frequency */
static const double param_tau = 3.;
/* ratio of coefficients of temporal vs spatial correlation */
static const double param_r02 = 0.01;
/* ratio of coefficients of major and minor axes of spatial correlation */
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
    /* return param_lam; */
  
  double r = sqrt(x1 * x1 + x2 * x2);
  
  if (r >= param_rct)
    return param_lam * r;
  else
    return cutoff(r, param_rct, param_lam * param_rct,
		  param_lam, 0.8 * param_lam * param_rct);
}

static double corr_time(double x0, double x1, double x2)
{
  /* return param_tau; */
  
  double r = sqrt(x1 * x1 + x2 * x2);

  if (r >= param_rct)
    return 2. * M_PI * param_tau / fabs( w_keplerian(x0, x1, x2) );
  else
    return cutoff(r, param_rct,
		  2. * M_PI * param_tau * pow(param_rct, 1.5),
		  2. * M_PI * param_tau * 1.5 * sqrt(param_rct),
		  0.8 * 2. * M_PI * param_tau * pow(param_rct, 1.5) );
}

static void set_velocity(double* v, double x0, double x1, double x2)
{
  double omega = w_keplerian(x0, x1, x2);
  v[0] = 0.;
  v[1] = -x2 * omega;
  v[2] = x1 * omega;

  /* v[1] = sin( x2 * 2. * M_PI / (param_x2end - param_x2start) ); */
  /* v[2] = 0.; */
}

/* unit vector in direction of spacetime correlation */
static void set_u0(double* u0, double x0, double x1, double x2)
{
  double psi, theta;

  double v[3];
  set_velocity(v, x0, x1, x2);

  psi = atan( sqrt(v[1] * v[1] + v[2] * v[2]) );

  if (v[1] == 0. && v[2] == 0.)
    theta = 0.;
  else
    theta = atan2(-v[2], -v[1]);
  
  u0[0] = cos(psi);
  u0[1] = cos(theta) * sin(psi);
  u0[2] = sin(theta) * sin(psi);
}

/* unit vector in direction of spatial correlation */
static void set_u1(double* u1, double x0, double x1, double x2)
{
  double theta;

  if (x1 == 0 && x2 == 0) {
    u1[0] = 0.;
    u1[1] = 0.;
    u1[2] = 0.;
  }
  else {
    theta = atan2(x2, x1) +
      copysign( -M_PI / 2. + M_PI / 9., w_keplerian(x0, x1, x2) );
    /* theta = atan2(1, dx0 * 2. * M_PI * */
    /* 		  cos(x2 * 2. * M_PI / (param_x2end - param_x2start) ) */
    /* 		  / (param_x2end - param_x2start) ); */
    
    u1[0] = 0.;
    u1[1] = cos(theta);
    u1[2] = sin(theta);
  }
}

static void set_h(double h[3][3], double x0, double x1, double x2)
{
  int i, j;
  double u0[3], u1[3];

  set_u0(u0, x0, x1, x2);
  set_u1(u1, x0, x1, x2);
  
  double gamm0, gamm1, beta0, beta1;

  gamm0 = param_r02 * corr_time(x0, x1, x2);
  beta0 = (1. - param_r02) * corr_time(x0, x1, x2);
  gamm1 = param_r12 * corr_length(x0, x1, x2);
  beta1 = (1. - param_r12) * corr_length(x0, x1, x2);  

  for (i = 0; i < 3; i++) {
    for (j = 0; j < 3; j++) {
      h[i][j] = beta0 * u0[i] * u0[j] + beta1 * u1[i] * u1[j];
      if (i == j)
	h[i][j] += gamm0 + (i == 0 ? 0. : gamm1);
    }
  }
}

/* dh[0][2][1] = dh[0][2]/dx[1] */
static void set_dh(double dh[][3][3], double x0, double x1, double x2,
		   double dx0, double dx1, double dx2)
{
  int i, j, k;

  double dx[3] = {dx0, dx1, dx2};
  
  /* hm[0][2][1] = h(x0 - dx0, x1, x2)[2][1] */
  double hm[3][3][3], hp[3][3][3]; 
  set_h(hm[0], x0 - dx0, x1, x2);
  set_h(hp[0], x0 + dx0, x1, x2);
  set_h(hm[1], x0, x1 - dx1, x2);
  set_h(hp[1], x0, x1 + dx1, x2);
  set_h(hm[2], x0, x1, x2 - dx2);
  set_h(hp[2], x0, x1, x2 + dx2);

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

void param_coeff(double* coeff, double x0, double x1, double x2, double dx0,
		  double dx1, double dx2, int index)
{
  double h[3][3], dh[3][3][3];
  set_h(h, x0, x1, x2);
  set_dh(dh, x0, x1, x2, dx0, dx1, dx2);
  
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
    
    values[i] = /* pow(r, 1.5) * */ gsl_ran_gaussian_ziggurat(rstate, 1.);
  }
}
