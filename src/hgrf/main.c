/*
  Compile with:   make all
                  or make (insert model name) e.g. make disk
  
  Sample run:     mpirun -np 4 disk -n 32 -solver 0 -v 1 1
                  mpiexec -n 4 ./poisson -n 32 -solver 0 -v 1 1
  
  To see options: use option help, -help, or --help

  Index naming conventions:
  x0 = t, x1 = r, x2 = phi
  Given a stencil {i,j,k}, HYPRE has k as the slowest varying index. 
  Thus, the indices correspond to i = x2, j = x1, k = x0. 
*/
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "_hypre_utilities.h"
#include "HYPRE_struct_ls.h"
#include "hdf5_utils.h"
#include "param.h"
#include "model.h"

// TODO generalize dimension, set dimension in model
void c_init_mpi(){
  /* Finalize MPI */
  int flag;
  MPI_Initialized(&flag);
  if (!flag)
    MPI_Init(NULL, NULL);
}

void c_end_mpi(){
      /* Finalize MPI */
  MPI_Finalize();
}

int c_main(int nk, int ni, int nj, int solver_id, int maxiter, int verbose, double param_r12,
           double* source, double spatial_angle_image[ni][nj], double velocity[ni][nj],
           double correlation_time_image[ni][nj], double correlation_length_image[ni][nj], double* output_video, double* values)
{
  int i, j, k;

  int myid, num_procs;

  int pi, pj, pk, npi, npj, npk;
  double dx0, dx1, dx2;
  int ilower[3], iupper[3];


  int n_pre, n_post;

  clock_t start_t = clock();
  clock_t check_t;

  HYPRE_StructGrid    grid;
  HYPRE_StructStencil stencil;
  HYPRE_StructMatrix  A;
  HYPRE_StructVector  b;
  HYPRE_StructVector  x;
  HYPRE_StructSolver  solver;
  HYPRE_StructSolver  precond;

  int num_iterations;
  double final_res_norm;

  int output, timer;

  char* dir_ptr;

  /* Initialize MPI */
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  /* Set defaults */
  npi = 1;
  npj = 1;
  npk = 1;                      /* default processor grid is 1 x 1 x N */
  n_pre  = 1;
  n_post = 1;
  output = 1;                  /* output data by default */
  timer  = 0;
  char* default_dir = ".";     /* output in current directory by default */
  dir_ptr = default_dir;


  /* Set dx0, dx1, dx2 */
  model_set_spacing(&dx0, &dx1, &dx2, ni, nj, nk, npi, npj, npk);

  /* Figure out processor grid (npi x npj x npk). Processor position
     indicated by pi, pj, pk. Size of local grid on processor is
     (ni x nj x nk) */

  pk = myid / (npi * npj);
  pj = (myid - pk * npi * npj) / npi;
  pi = myid - pk * npi * npj - pj * npi;

  /* Figure out the extents of each processor's piece of the grid. */
  ilower[0] = pi * ni;
  ilower[1] = pj * nj;
  ilower[2] = pk * nk;

  iupper[0] = ilower[0] + ni - 1;
  iupper[1] = ilower[1] + nj - 1;
  iupper[2] = ilower[2] + nk - 1;

  /* Set up and assemble grid */
  {
    HYPRE_StructGridCreate(MPI_COMM_WORLD, 3, &grid);
    HYPRE_StructGridSetExtents(grid, ilower, iupper);

    /* Set periodic boundary conditions of model */
    int bound_con[3];
    model_set_periodic(bound_con, ni, nj, nk, npi, npj, npk, 3);
    HYPRE_StructGridSetPeriodic(grid, bound_con);

    HYPRE_StructGridAssemble(grid);
  }

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Grid initialized: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);

  /* Initialize stencil and Struct Matrix, and set stencil values */
  model_create_stencil(&stencil, 3);

  HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &A);
  HYPRE_StructMatrixInitialize(A);

  model_set_stencil_values(&A, ilower, iupper, ni, nj, nk, pi, pj, pk, dx0, dx1, dx2, param_r12,
			               spatial_angle_image, velocity, correlation_time_image, correlation_length_image, values);

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Stencils values set: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);

  /* Fix boundary conditions and assemble Struct Matrix */
  model_set_bound(&A, ni, nj, nk, pi, pj, pk, npi, npj, npk, dx0, dx1, dx2);

  HYPRE_StructMatrixAssemble(A);

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Boundary conditions set: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);

  /* Set up Struct Vectors for b and x */
  {
    int    nvalues = ni * nj * nk;


    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

    HYPRE_StructVectorInitialize(b);
    HYPRE_StructVectorInitialize(x);

    HYPRE_StructVectorSetBoxValues(b, ilower, iupper, source);

    for (i = 0; i < nvalues; i ++)
      source[i] = 0.0;
    HYPRE_StructVectorSetBoxValues(x, ilower, iupper, source);

    HYPRE_StructVectorAssemble(b);
    HYPRE_StructVectorAssemble(x);
  }

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Struct vector assembled: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);

  /* Set up and use a struct solver */
  if (solver_id == 0) {
    HYPRE_StructPCGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructPCGSetMaxIter(solver, maxiter);
    HYPRE_StructPCGSetTol(solver, 1.0e-06 );
    HYPRE_StructPCGSetTwoNorm(solver, 1 );
    HYPRE_StructPCGSetRelChange(solver, 0 );
    HYPRE_StructPCGSetPrintLevel(solver, verbose ); /* print each CG iteration */
    HYPRE_StructPCGSetLogging(solver, 1);

    /* Use symmetric SMG as preconditioner */
    HYPRE_StructSMGCreate(MPI_COMM_WORLD, &precond);
    HYPRE_StructSMGSetMemoryUse(precond, 0);
    HYPRE_StructSMGSetMaxIter(precond, 1);
    HYPRE_StructSMGSetTol(precond, 0.0);
    HYPRE_StructSMGSetZeroGuess(precond);
    HYPRE_StructSMGSetNumPreRelax(precond, 1);
    HYPRE_StructSMGSetNumPostRelax(precond, 1);

    /* Set the preconditioner and solve */
    HYPRE_StructPCGSetPrecond(solver, HYPRE_StructSMGSolve,
			      HYPRE_StructSMGSetup, precond);
    HYPRE_StructPCGSetup(solver, A, b, x);
    HYPRE_StructPCGSolve(solver, A, b, x);

    /* Get some info on the run */
    HYPRE_StructPCGGetNumIterations(solver, &num_iterations);
    HYPRE_StructPCGGetFinalRelativeResidualNorm(solver, &final_res_norm);

    /* Clean up */
    HYPRE_StructPCGDestroy(solver);
  }

  if (solver_id == 1) {
    HYPRE_StructSMGCreate(MPI_COMM_WORLD, &solver);
    HYPRE_StructSMGSetMemoryUse(solver, 0);
    HYPRE_StructSMGSetMaxIter(solver, maxiter);
    HYPRE_StructSMGSetTol(solver, 1.0e-06);
    HYPRE_StructSMGSetRelChange(solver, 0);
    HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
    HYPRE_StructSMGSetNumPostRelax(solver, n_post);
    /* Logging must be on to get iterations and residual norm info below */
    // TODO fix SMG logging
    HYPRE_StructSMGSetPrintLevel(solver, verbose);
    HYPRE_StructSMGSetLogging(solver, verbose);

    /* Setup and solve */
    HYPRE_StructSMGSetup(solver, A, b, x);
    HYPRE_StructSMGSolve(solver, A, b, x);

    /* Get some info on the run */
    HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
    HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

    /* Clean up */
    HYPRE_StructSMGDestroy(solver);
  }

  /* Matrix - vector multiplication */
  if (solver_id == 2) {
    HYPRE_StructMatrixMatvec(1.0, A, b, 0.0, x);
  }


  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Solver finished: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);

  if ( (myid == 0) && (verbose) ) {
    printf("\n");
    printf("Iterations = %d\n", num_iterations);
    printf("Final Relative Residual Norm = %g\n", final_res_norm);
    printf("\n");
  }

  int nvalues = ni * nj * nk;

  HYPRE_StructVectorGetBoxValues(x, ilower, iupper, output_video);

  /* Free HYPRE memory */
  HYPRE_StructGridDestroy(grid);
  HYPRE_StructStencilDestroy(stencil);
  HYPRE_StructMatrixDestroy(A);
  HYPRE_StructVectorDestroy(b);
  HYPRE_StructVectorDestroy(x);

  return (0);
}
