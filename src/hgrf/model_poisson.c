#include "model.h"

#include <math.h>
#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"

#include "param.h"

#define NSTENCIL 7

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

void model_set_periodic(int* bound, int ni, int nj, int nk,
			int npi, int npj, int npk, int dim)
{
  bound[0] = npi * ni;
  bound[1] = npj * nj;
  bound[2] = npk * nk;
}

/* Recall i = x2, j = x1, k = x0 */
/*
  22 14 21    10 04 09    26 18 25    ^
  11 05 12    01 00 02    15 06 16    |
  19 13 20    07 03 08    23 17 24    j i ->    k - - >
  
  Delete zero entries:
  xx xx xx    xx 04 xx    xx xx xx    ^
  xx 05 xx    01 00 02    xx 06 xx    |			 
  xx xx xx    xx 03 xx    xx xx xx    j i ->    k - - >			
*/
void model_create_stencil(HYPRE_StructStencil* stencil, int dim)
{
  HYPRE_StructStencilCreate(3, NSTENCIL, stencil);

  int entry;
  int offsets[NSTENCIL][3] = {{0,0,0},
			      {-1,0,0}, {1,0,0}, 
			      {0,-1,0}, {0,1,0}, 
			      {0,0,-1}, {0,0,1} 
			      /* {-1,-1,0}, {1,-1,0},  */
			      /* {1,1,0}, {-1,1,0},  */
			      /* {-1,0,-1}, {1,0,-1},  */
			      /* {0,-1,-1}, {0,1,-1},  */
			      /* {-1,0,1}, {1,0,1},  */
			      /* {0,-1,1}, {0,1,1}, */
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
    double coeff[4];
    
    param_coeff(coeff, 0., 0., 0., dx0, dx1, dx2, 4);

    /*
      xx xx xx    xx 04 xx    xx xx xx    ^
      xx 05 xx    01 00 02    xx 06 xx    |			 
      xx xx xx    xx 03 xx    xx xx xx    j i ->    k - - >			
    */
        
    values[i]    = coeff[3];
    values[i+1]  = coeff[0];
    values[i+2]  = coeff[0];
    values[i+3]  = coeff[1];
    values[i+4]  = coeff[1];
    values[i+5]  = coeff[2];
    values[i+6]  = coeff[2];

    /* values[i+7]  = 0.; */
    /* values[i+8]  = 0.; */
    /* values[i+9]  = 0.; */
    /* values[i+10] = 0.; */
    /* values[i+11] = 0.; */
    /* values[i+12] = 0.; */
    /* values[i+13] = 0.; */
    /* values[i+14] = 0.; */
    /* values[i+15] = 0.; */
    /* values[i+16] = 0.; */
    /* values[i+17] = 0.; */
    /* values[i+18] = 0.; */
    
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
  /* int j; */
  /* int bc_ilower[3]; */
  /* int bc_iupper[3]; */
  /* int nentries = 1; */
  /* int nvalues  = nentries * ni * nk; /\* number of stencil entries times the  */
  /* 					length of one side of my grid box *\/ */
  /* double *values; */
  /* int stencil_indices[nentries]; */
  /* values = (double*) calloc(nvalues, sizeof(double)); */

  /* for (j = 0; j < nvalues; j++) */
  /*   values[j] = 0.; */
  
  /* /\* */
  /*   xx xx xx    xx 04 xx    xx xx xx    ^ */
  /*   xx 05 xx    01 00 02    xx 06 xx    |			  */
  /*   xx xx xx    xx 03 xx    xx xx xx    j i ->    k - - >			 */
  /* *\/ */
  
  /* /\* Recall: pi, pj, and pk describe position in the processor grid *\/     */
  /* if (pi == 0) { */
  /*   /\* Left grid points *\/     */
  /*   bc_ilower[0] = pi * ni; */
  /*   bc_ilower[1] = pj * nj; */
  /*   bc_ilower[2] = pk * nk; */
    
  /*   bc_iupper[0] = bc_ilower[0]; */
  /*   bc_iupper[1] = bc_ilower[1] + nj - 1; */
  /*   bc_iupper[2] = bc_ilower[2] + nk - 1; */
    
  /*   stencil_indices[0] = 1; */
    
  /*   HYPRE_StructMatrixSetBoxValues(*A, bc_ilower, bc_iupper, nentries, */
  /* 				   stencil_indices, values); */
  /* } */
  
  /* if (pi == npi - 1) { */
  /*   /\* Right grid points *\/ */
  /*   bc_ilower[0] = pi * ni + ni - 1; */
  /*   bc_ilower[1] = pj * nj; */
  /*   bc_ilower[2] = pk * nk; */
    
  /*   bc_iupper[0] = bc_ilower[0]; */
  /*   bc_iupper[1] = bc_ilower[1] + nj - 1; */
  /*   bc_iupper[2] = bc_ilower[2] + nk - 1; */
    
  /*   stencil_indices[0] = 2; */
    
  /*   HYPRE_StructMatrixSetBoxValues(*A, bc_ilower, bc_iupper, nentries, */
  /* 				   stencil_indices, values); */
  /* } */
  
  /* if (pj == 0) { */
  /*   /\* Bottom grid points *\/     */
  /*   bc_ilower[0] = pi * ni; */
  /*   bc_ilower[1] = pj * nj; */
  /*   bc_ilower[2] = pk * nk; */
    
  /*   bc_iupper[0] = bc_ilower[0] + ni - 1; */
  /*   bc_iupper[1] = bc_ilower[1]; */
  /*   bc_iupper[2] = bc_ilower[2] + nk - 1; */
    
  /*   stencil_indices[0] = 3; */
    
  /*   HYPRE_StructMatrixSetBoxValues(*A, bc_ilower, bc_iupper, nentries, */
  /* 				   stencil_indices, values); */
  /* } */
  
  /* if (pj == npj - 1) { */
  /*   /\* Upper grid points *\/ */
  /*   bc_ilower[0] = pi * ni; */
  /*   bc_ilower[1] = pj * nj + nj - 1; */
  /*   bc_ilower[2] = pk * nk; */
    
  /*   bc_iupper[0] = bc_ilower[0] + ni - 1; */
  /*   bc_iupper[1] = bc_ilower[1]; */
  /*   bc_iupper[2] = bc_ilower[2] + nk - 1; */
    
  /*   stencil_indices[0] = 4; */
    
  /*   HYPRE_StructMatrixSetBoxValues(*A, bc_ilower, bc_iupper, nentries, */
  /* 				   stencil_indices, values); */
  /* } */
  
  /* if (pk == 0) { */
  /*   /\* Front grid points *\/     */
  /*   bc_ilower[0] = pi * ni; */
  /*   bc_ilower[1] = pj * nj; */
  /*   bc_ilower[2] = pk * nk; */
    
  /*   bc_iupper[0] = bc_ilower[0] + ni - 1; */
  /*   bc_iupper[1] = bc_ilower[1] + nj - 1; */
  /*   bc_iupper[2] = bc_ilower[2]; */
    
  /*   stencil_indices[0] = 5; */
    
  /*   HYPRE_StructMatrixSetBoxValues(*A, bc_ilower, bc_iupper, nentries, */
  /* 				   stencil_indices, values); */
  /* } */
  
  /* if (pk == npk - 1) { */
  /*   /\* Back grid points *\/ */
  /*   bc_ilower[0] = pi * ni; */
  /*   bc_ilower[1] = pj * nj; */
  /*   bc_ilower[2] = pk * nk + nk - 1; */
    
  /*   bc_iupper[0] = bc_ilower[0] + ni - 1; */
  /*   bc_iupper[1] = bc_ilower[1] + nj - 1; */
  /*   bc_iupper[2] = bc_ilower[2]; */
    
  /*   stencil_indices[0] = 6; */
    
  /*   HYPRE_StructMatrixSetBoxValues(*A, bc_ilower, bc_iupper, nentries, */
  /* 				   stencil_indices, values); */
  /* } */
  
  /* free(values); */
}

double model_area(int i, int j, int k, int ni, int nj, int nk,
		  int pi, int pj, int pk, double dx0, double dx1, double dx2)
{
  return dx0 * dx1 * dx2;
}
