#ifndef PARAM
#define PARAM

#include <gsl/gsl_rng.h>

extern const char model_name[];

extern const double param_mass;
extern const double param_mdot;
extern const double param_x0start;
extern const double param_x0end;
extern const double param_x1start;
extern const double param_x1end;
extern const double param_x2start;
extern const double param_x2end;


double param_env(double raw, double avg_raw, double var_raw,
		 int i, int j, int k, int ni, int nj, int nk,
		 int pi, int pj, int pk, double dx0, double dx1, double dx2, double param_rct);

void param_coeff(double* coeff, double x0, double x1, double x2, double dx0, double dx1, double dx2,
                 double param_rct, double param_r12, double param_r02, double spatial_angle,
                 double correlation_time, double correlation_length);

void param_set_source(double* values, gsl_rng* rstate, int ni, int nj, int nk,
		      int pi, int pj, int pk, int npi, int npj, int npk,
		      double dx0, double dx1, double dx2);

#endif
