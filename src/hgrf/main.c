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

int c_main ()
{
  int i, j, k;

  int myid, num_procs;

  int ni, nj, nk, pi, pj, pk, npi, npj, npk;
  double dx0, dx1, dx2;
  int ilower[3], iupper[3];

  int solver_id;
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

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  /* Set defaults */
  ni  = 64;                    /* nj and nk are set equal to ni later */
  nj  = 64;
  nk  = 100;
  npi = 1;
  npj = 1;
  npk = num_procs;             /* default processor grid is 1 x 1 x N */
  solver_id = 0;
  n_pre  = 1;
  n_post = 1;
  output = 1;                  /* output data by default */
  timer  = 0;
  char* default_dir = ".";     /* output in current directory by default */
  dir_ptr = default_dir;

  /* Initiialize rng */
  const gsl_rng_type *T;
  gsl_rng *rstate;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  rstate = gsl_rng_alloc(T);
  gsl_rng_set(rstate, model_set_gsl_seed(gsl_rng_default_seed, myid));


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

  model_set_stencil_values(&A, ilower, iupper, ni, nj, nk, pi, pj, pk,
			   dx0, dx1, dx2);

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
    double *values;

    values = (double*) calloc(nvalues, sizeof(double));

    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &b);
    HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &x);

    HYPRE_StructVectorInitialize(b);
    HYPRE_StructVectorInitialize(x);

    /* Set the values */
    param_set_source(values, rstate, ni, nj, nk, pi, pj, pk, npi, npj, npk,
		     dx0, dx1, dx2);

    HYPRE_StructVectorSetBoxValues(b, ilower, iupper, values);

    for (i = 0; i < nvalues; i ++)
      values[i] = 0.0;
    HYPRE_StructVectorSetBoxValues(x, ilower, iupper, values);

    free(values);

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
    HYPRE_StructPCGSetMaxIter(solver, 50 );
    HYPRE_StructPCGSetTol(solver, 1.0e-06 );
    HYPRE_StructPCGSetTwoNorm(solver, 1 );
    HYPRE_StructPCGSetRelChange(solver, 0 );
    HYPRE_StructPCGSetPrintLevel(solver, 2 ); /* print each CG iteration */
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
    HYPRE_StructSMGSetMaxIter(solver, 50);
    HYPRE_StructSMGSetTol(solver, 1.0e-06);
    HYPRE_StructSMGSetRelChange(solver, 0);
    HYPRE_StructSMGSetNumPreRelax(solver, n_pre);
    HYPRE_StructSMGSetNumPostRelax(solver, n_post);
    /* Logging must be on to get iterations and residual norm info below */
    // TODO fix SMG logging
    HYPRE_StructSMGSetPrintLevel(solver, 2);
    HYPRE_StructSMGSetLogging(solver, 1);

    /* Setup and solve */
    HYPRE_StructSMGSetup(solver, A, b, x);
    HYPRE_StructSMGSolve(solver, A, b, x);

    /* Get some info on the run */
    HYPRE_StructSMGGetNumIterations(solver, &num_iterations);
    HYPRE_StructSMGGetFinalRelativeResidualNorm(solver, &final_res_norm);

    /* Clean up */
    HYPRE_StructSMGDestroy(solver);
  }

  check_t = clock();
  if ( (myid == 0) && (timer) )
    printf("Solver finished: t = %lf\n\n",
	   (double)(check_t - start_t) / CLOCKS_PER_SEC);

  if (myid == 0) {
    printf("\n");
    printf("Iterations = %d\n", num_iterations);
    printf("Final Relative Residual Norm = %g\n", final_res_norm);
    printf("\n");
  }

  /* Output data */
  if (output) {
    /* Get the local raw data */
    int nvalues = ni * nj * nk;

    double *raw = (double*)calloc(nvalues, sizeof(double));
    double *env = (double*)calloc(nvalues, sizeof(double));

    HYPRE_StructVectorGetBoxValues(x, ilower, iupper, raw);

    /* Find statistics for raw data and envelope */
    double avg_raw = 0.;
    double avg_env = 0.;
    double var_raw = 0.;
    double var_env = 0.;
    double min_raw = raw[0];
    double max_raw = raw[0];
    double min_env;
    double max_env;

    double *lc_raw = (double*)calloc(npk * nk, sizeof(double));
    double *lc_env = (double*)calloc(npk * nk, sizeof(double));

    /* Calculate mean and variance for raw data */
    {
      for(i = 0; i < nvalues; i++)
	avg_raw += raw[i];

      MPI_Allreduce(MPI_IN_PLACE, &avg_raw, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);

      avg_raw /= (npi * npj * npk * nvalues);

      /* Second pass for variance */
      for (i = 0; i < nvalues; i++)
	var_raw += (raw[i] - avg_raw) * (raw[i] - avg_raw);

      MPI_Allreduce(MPI_IN_PLACE, &var_raw, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);

      var_raw /= (npi * npj * npk * nvalues - 1);
    }

    /* Add envelope and calculate envelope average */
    {
      int l = 0;
      for (k = 0; k < nk; k++) {
	for (j = 0; j < nj; j++) {
	  for (i = 0; i < ni; i++) {
	    env[l] = param_env(raw[l], avg_raw, var_raw, i, j, k, ni, nj, nk,
			       pi, pj, pk, dx0, dx1, dx2);

	    avg_env += env[l];

	    l++;
	  }
	}
      }

      MPI_Allreduce(MPI_IN_PLACE, &avg_env, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);

      avg_env /= (npi * npj * npk * nvalues);
    }

    min_env = env[0];
    max_env = env[0];

    /* Find min, max, lightcurve, var_env */
    {
      int l = 0;
      double area;
      for (k = pk * nk; k < (pk + 1) * nk; k++) {
	for (j = 0; j < nj; j++) {
	  for (i = 0; i < ni; i++) {
	    area = model_area(i, j, k, ni, nj, nk, pi, pj, pk, dx0, dx1, dx2);

	    lc_raw[k] += raw[l] * area;
	    lc_env[k] += env[l] * area;

	    if (env[l] < min_env)
	      min_env = env[l];
	    if (env[l] > max_env)
	      max_env = env[l];

	    var_env += (env[l] - avg_env) * (env[l] - avg_env);

	    l++;
	  }
	}
      }

      MPI_Allreduce(MPI_IN_PLACE, &var_env, 1, MPI_DOUBLE,
		    MPI_SUM, MPI_COMM_WORLD);

      var_env /= (npi * npj * npk * nvalues - 1);
    }

    // TODO turn allreduce into reduce for min, max, and var
    // (currently required for hdf5_write_single_val)
    MPI_Allreduce(MPI_IN_PLACE, &min_raw, 1, MPI_DOUBLE,
		  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_raw, 1, MPI_DOUBLE,
		  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &min_env, 1, MPI_DOUBLE,
		  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_env, 1, MPI_DOUBLE,
		  MPI_MAX, MPI_COMM_WORLD);
    if (myid == 0) {
      MPI_Reduce(MPI_IN_PLACE, lc_raw, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, lc_env, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else {
      MPI_Reduce(lc_raw, lc_raw, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(lc_env, lc_env, npk * nk, MPI_DOUBLE,
		 MPI_SUM, 0, MPI_COMM_WORLD);
    }

    /* File i/o */
    char filename[255];

    if (myid == 0) {
      time_t rawtime;
      struct tm * timeinfo;
      char buffer[255];

      time(&rawtime);
      timeinfo = localtime(&rawtime);
      strftime(buffer, 255, "%Y_%m_%d_%H%M%S", timeinfo);

      sprintf(filename, "%s/%s_%d_%d_%d_%s.h5", dir_ptr, model_name,
	      npi * ni, npj * nj, npk * nk, buffer);

      printf("%s\n\n", filename);
    }

    MPI_Bcast(&filename, 255, MPI_CHAR, 0, MPI_COMM_WORLD);
    hdf5_create(filename);

    /* Save solution to output file*/
    hdf5_set_directory("/");
    hdf5_make_directory("data");
    hdf5_set_directory("/data/");
    // TODO create further heirarchy in file structure

    /* note: HYPRE has k as the slowest varying, opposite of HDF5 */
    {
      hsize_t fdims[3]  = {npk * nk, npj * nj, npi * ni};
      hsize_t fstart[3] = {pk * nk, pj * nj, pi * ni};
      hsize_t fcount[3] = {nk, nj, ni};
      hsize_t mdims[3]  = {nk, nj, ni};
      hsize_t mstart[3] = {0, 0, 0};

      hdf5_write_array(raw, "data_raw", 3, fdims, fstart, fcount,
		       mdims, mstart, H5T_NATIVE_DOUBLE);
      hdf5_write_array(env, "data_env", 3, fdims, fstart, fcount,
		       mdims, mstart, H5T_NATIVE_DOUBLE);
    }

    /* Output lightcurve and parameters */
    {
      hsize_t fdims  = npk * nk;
      hsize_t fstart = 0;
      hsize_t fcount = 0;
      hsize_t mdims  = 0;
      hsize_t mstart = 0;

      if (myid == 0) {
	fcount = npk * nk;
	mdims  = npk * nk;
      }

      hdf5_write_array(lc_raw, "lc_raw", 1, &fdims, &fstart, &fcount,
		       &mdims, &mstart, H5T_NATIVE_DOUBLE);
      hdf5_write_array(lc_env, "lc_env", 1, &fdims, &fstart, &fcount,
		       &mdims, &mstart, H5T_NATIVE_DOUBLE);
    }

    hdf5_set_directory("/");
    hdf5_make_directory("params");
    hdf5_set_directory("/params/");

    hdf5_write_single_val(&param_mass, "mass", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_mdot, "mdot", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x0start, "x0start", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x0end, "x0end", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x1start, "x1start", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x1end, "x1end", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x2start, "x2start", H5T_IEEE_F64LE);
    hdf5_write_single_val(&param_x2end, "x2end", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx0, "dx0", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx1, "dx1", H5T_IEEE_F64LE);
    hdf5_write_single_val(&dx2, "dx2", H5T_IEEE_F64LE);

    hdf5_write_single_val(&npi, "npi", H5T_STD_I32LE);
    hdf5_write_single_val(&npj, "npj", H5T_STD_I32LE);
    hdf5_write_single_val(&npk, "npk", H5T_STD_I32LE);
    hdf5_write_single_val(&ni, "ni", H5T_STD_I32LE);
    hdf5_write_single_val(&nj, "nj", H5T_STD_I32LE);
    hdf5_write_single_val(&nk, "nk", H5T_STD_I32LE);
    hdf5_write_single_val(&gsl_rng_default_seed, "seed", H5T_STD_U64LE);

    hdf5_set_directory("/");
    hdf5_make_directory("stats");
    hdf5_set_directory("/stats/");
    hdf5_write_single_val(&min_raw, "min_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&max_raw, "max_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&avg_raw, "avg_raw", H5T_IEEE_F64LE);
    hdf5_write_single_val(&var_raw, "var_raw", H5T_IEEE_F64LE);

    hdf5_write_single_val(&min_env, "min_env", H5T_IEEE_F64LE);
    hdf5_write_single_val(&max_env, "max_env", H5T_IEEE_F64LE);
    hdf5_write_single_val(&avg_env, "avg_env", H5T_IEEE_F64LE);
    hdf5_write_single_val(&var_env, "var_env", H5T_IEEE_F64LE);

    hdf5_close();

    check_t = clock();
    if ( (myid == 0) && (timer) )
      printf("Data output: t = %lf\n\n",
	     (double)(check_t - start_t) / CLOCKS_PER_SEC);

    free(raw);
    free(env);
    free(lc_raw);
    free(lc_env);
  }

  /* Free HYPRE memory */
  HYPRE_StructGridDestroy(grid);
  HYPRE_StructStencilDestroy(stencil);
  HYPRE_StructMatrixDestroy(A);
  HYPRE_StructVectorDestroy(b);
  HYPRE_StructVectorDestroy(x);

  /* Free GSL rng state */
  gsl_rng_free(rstate);

  /* Finalize MPI */
  MPI_Finalize();

  return (0);
}
