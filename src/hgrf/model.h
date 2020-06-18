#ifndef MODEL
#define MODEL

#include "HYPRE_struct_ls.h"

int model_set_gsl_seed(int seed, int myid);

void model_set_spacing(double* dx0, double* dx1, double* dx2,
		       int ni, int nj, int nk, int npi, int npj, int npk);

void model_set_periodic(int* bound, int ni, int nj, int nk,
			int npi, int npj, int npk, int dim);

void model_create_stencil(HYPRE_StructStencil* stencil, int dim);

void model_set_stencil_values(HYPRE_StructMatrix* A, int* ilower, int* iupper,
			      int ni, int nj, int nk, int pi, int pj, int pk,
			      double dx0, double dx1, double dx2);

void model_set_bound(HYPRE_StructMatrix* A, int ni, int nj, int nk,
		     int pi, int pj, int pk, int npi, int npj, int npk,
		     double dx0, double dx1, double dx2);

double model_area(int i, int j, int k, int ni, int nj, int nk,
		  int pi, int pj, int pk, double dx0, double dx1, double dx2);

#endif
