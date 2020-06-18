#include "model.h"

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"

#include "param.h"

#define NSTENCIL 19
// TODO generalize dimensions in main, set dimension in model

int model_set_gsl_seed(int seed, int myid)
{
  return seed + myid;
}

void model_set_spacing(double* dx0, double* dx1, double* dx2,
		       int ni, int nj, int nk, int npi, int npj, int npk)
{
  *dx0 = (param_x0end - param_x0start) / (npk * nk);
  *dx1 = (param_x1end - param_x1start) / (npj * nj); 
  *dx2 = (param_x2end - param_x2start) / (npi * ni);
}

/* Periodic in phi and t */
void model_set_periodic(int* bound, int ni, int nj, int nk,
			int npi, int npj, int npk, int dim)
{
  bound[0] = npi * ni;
  bound[1] = 0;
  bound[2] = npk * nk;
}

/* Recall i = x2, j = x1, k = x0 */
/*
  22 14 21    10 04 09    26 18 25    ^
  11 05 12    01 00 02    15 06 16    |
  19 13 20    07 03 08    23 17 24    j i ->    k - - >
  
  Delete zero entries:
  xx 14 xx    10 04 09    xx 18 xx    ^
  11 05 12    01 00 02    15 06 16    |			 
  xx 13 xx    07 03 08    xx 17 xx    j i ->    k - - >			
*/
void model_create_stencil(HYPRE_StructStencil* stencil, int dim)
{
  HYPRE_StructStencilCreate(3, NSTENCIL, stencil);

  int entry;
  int offsets[NSTENCIL][3] = {{0,0,0},
			      {-1,0,0}, {1,0,0}, 
			      {0,-1,0}, {0,1,0}, 
			      {0,0,-1}, {0,0,1}, 
			      {-1,-1,0}, {1,-1,0}, 
			      {1,1,0}, {-1,1,0}, 
			      {-1,0,-1}, {1,0,-1}, 
			      {0,-1,-1}, {0,1,-1}, 
			      {-1,0,1}, {1,0,1}, 
			      {0,-1,1}, {0,1,1}
			      /* {-1,-1,-1}, {1,-1,-1}, */
			      /* {1,1,-1}, {-1,1,-1}, */
			      /* {-1,-1,1}, {1,-1,1}, */
			      /* {1,1,1}, {-1,1,1} */
  };
  for (entry = 0; entry < NSTENCIL; entry++)
    HYPRE_StructStencilSetElement(*stencil, entry, offsets[entry]);
}

void model_set_stencil_values(HYPRE_StructMatrix* A, int* ilower, int* iupper,
			      int ni, int nj, int nk, int pi, int pj, int pk,
			      double dx0, double dx1, double dx2)
{
  int i, j;
  int nentries = NSTENCIL;
  int nvalues = nentries * ni * nj * nk;
  double *values;
  int stencil_indices[NSTENCIL];
  
  values = (double*) calloc(nvalues, sizeof(double));
  
  for (j = 0; j < nentries; j++)
    stencil_indices[j] = j;
  
  for (i = 0; i < nvalues; i += nentries) {
    double x0, x1, x2;
    double coeff[6];
    int gridi, gridj, gridk, temp;
    
    temp = i / nentries;
    gridk = temp / (ni * nj);
    gridj = (temp - ni * nj * gridk) / ni;
    gridi = temp - ni * nj * gridk + (pi - gridj) * ni;
    gridj += pj * nj;
    gridk += pk * nk;
    
    x0 = param_x0start + dx0 * gridk;			
    x1 = param_x1start + dx1 * gridj;
    x2 = param_x2start + dx2 * gridi;
    
    param_coeff(coeff, x0, x1, x2, dx0, dx1, dx2, 6);
    
    /*0=a, 1=b, 2=c, 3=d, 4=e, 5=f*/
    /*
      xx 14 xx    10 04 09    xx 18 xx    ^
      11 05 12    01 00 02    15 06 16    |			 
      xx 13 xx    07 03 08    xx 17 xx    j i ->    k - - >			
    */
    
    values[i]    = coeff[5];
    values[i+1]  = coeff[0];
    values[i+2]  = coeff[0];
    values[i+3]  = coeff[2];
    values[i+4]  = coeff[2];
    values[i+5]  = coeff[4];
    values[i+6]  = coeff[4];
    values[i+7]  = coeff[1];
    values[i+8]  = -coeff[1];
    values[i+9]  = coeff[1];
    values[i+10] = -coeff[1];
    values[i+11] = coeff[3];
    values[i+12] = -coeff[3];
    values[i+13] = 0.;
    values[i+14] = 0.;
    values[i+15] = -coeff[3];
    values[i+16] = coeff[3];
    values[i+17] = 0.;
    values[i+18] = 0.;
    
    /* values[i+19] = 0.; */
    /* values[i+20] = 0.; */
    /* values[i+21] = 0.; */
    /* values[i+22] = 0.; */
    /* values[i+23] = 0.; */
    /* values[i+24] = 0.; */
    /* values[i+25] = 0.; */
    /* values[i+26] = 0.; */
  }
  
  HYPRE_StructMatrixSetBoxValues(*A, ilower, iupper, nentries,
				 stencil_indices, values);
  
  free(values);
}

void model_set_bound(HYPRE_StructMatrix* A, int ni, int nj, int nk,
		     int pi, int pj, int pk, int npi, int npj, int npk,
		     double dx0, double dx1, double dx2)
{
  int j;
  int bc_ilower[3];
  int bc_iupper[3];
  int nentries = 6;
  int nvalues  = nentries * ni * nk; /* number of stencil entries times the 
					length of one side of my grid box */
  double *values;
  int stencil_indices[nentries];
  values = (double*) calloc(nvalues, sizeof(double));
  
  /*0=a, 1=b, 2=c, 3=d, 4=e, 5=f*/
  /*
    xx 14 xx    10 04 09    xx 18 xx    ^
    11 05 12    01 00 02    15 06 16    |			 
    xx 13 xx    07 03 08    xx 17 xx    j i ->    k - - >			
  */

  /* Recall: pi and pj describe position in the processor grid */    
  if (pj == 0) {
    /* Bottom grid points */
    double coeff[6];
    
    param_coeff(coeff, 0., param_x1start, 0., dx0, dx1, dx2, 6);
    
    for (j = 0; j < nvalues; j += nentries) {
      values[j]   = coeff[5] + coeff[2];
      values[j+1] = coeff[0] + coeff[1];
      values[j+2] = coeff[0] - coeff[1];
      values[j+3] = 0.0;
      values[j+4] = 0.0;
      values[j+5] = 0.0;
    }
    
    bc_ilower[0] = pi * ni;
    bc_ilower[1] = pj * nj;
    bc_ilower[2] = pk * nk;
    
    bc_iupper[0] = bc_ilower[0] + ni - 1;
    bc_iupper[1] = bc_ilower[1];
    bc_iupper[2] = bc_ilower[2] + nk - 1;
    
    stencil_indices[0] = 0;
    stencil_indices[1] = 1;
    stencil_indices[2] = 2;
    stencil_indices[3] = 3;
    stencil_indices[4] = 7;
    stencil_indices[5] = 8;
    
    HYPRE_StructMatrixSetBoxValues(*A, bc_ilower, bc_iupper, nentries,
				   stencil_indices, values);
  }
  
  if (pj == npj - 1) {
    /* Upper grid points */
    double coeff[6];
    
    bc_ilower[0] = pi * ni;
    bc_ilower[1] = pj * nj + nj - 1;
    bc_ilower[2] = pk * nk;
    
    bc_iupper[0] = bc_ilower[0] + ni - 1;
    bc_iupper[1] = bc_ilower[1];
    bc_iupper[2] = bc_ilower[2] + nk - 1;
    
    param_coeff(coeff, 0., param_x1end - dx1, 0., dx0, dx1, dx2, 6);
    
    for (j = 0; j < nvalues; j += nentries) {
      values[j]   = coeff[5] + coeff[2];
      values[j+1] = coeff[0] - coeff[1];
      values[j+2] = coeff[0] + coeff[1];
      values[j+3] = 0.0;
      values[j+4] = 0.0;
      values[j+5] = 0.0;
    }
    
    stencil_indices[0] = 0;
    stencil_indices[1] = 1;
    stencil_indices[2] = 2;
    stencil_indices[3] = 4;
    stencil_indices[4] = 9;
    stencil_indices[5] = 10;
    
    HYPRE_StructMatrixSetBoxValues(*A, bc_ilower, bc_iupper, nentries,
				   stencil_indices, values);
  }
  
  free(values);
}

double model_area(int i, int j, int k, int ni, int nj, int nk,
		  int pi, int pj, int pk, double dx0, double dx1, double dx2)
{
  return 0.5 * exp( 2. * (j + pj * nj) * dx1 ) *
    ( exp( 2. * dx1 ) - 1. ) * dx2;
}
