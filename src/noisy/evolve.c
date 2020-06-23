
/*

noisy:

Generate a gaussian random field in 2D with 
locally anisotropic correlation function, 
locally varying correlation time.

Follows the technique of 
Lindgren, Rue, and Lindstr\:om 2011, J.R. Statist. Soc. B 73, pp 423-498.
https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2011.00777.x
in particular, implements eq. 17, which has power spectrum given by eq. 18.

Based on work by 
David Daeyoung Lee 
Charles Gammie
on applications in disk turbulence.

CFG 22 Dec 2019

*/

#include "noisy.h"

/* grid functions necessary for diffusive and advective evolution */
void grid_function_calc(
    int nx, int ny,
    double F_coeff_gradx[nx][ny][4],
    double F_coeff_grady[nx][ny][4],
    double v[nx][ny][4][2],
    double T[nx][ny],
    double *Kmax,
    double *Vmax,
    double PARAM_RAT,
    double* principal_angle_image,
    double* advection_velocity_image,
    double* diffusion_coefficient_image,
    double* correlation_time_image
    ) {
    void principal_axis_from_angle(double *e1x, double *e1y, double *e2x, double *e2y, double principal_angle);

    /* preparatory work: calculate some grid functions */
    int i,j,k;
    double e1x,e1y,e2x,e2y;
    double K1,K2;
    double principal_angle;

    *Kmax = 0.;
    *Vmax = 0.;
    k=0;
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {

        if (i == nx-1) {
            principal_angle = principal_angle_image[k] + (principal_angle_image[k] - principal_angle_image[k-ny]) / 2.0;
            K1 = diffusion_coefficient_image[k] + (diffusion_coefficient_image[k] - diffusion_coefficient_image[k-ny]) / 2.0;
            v[i][j][0][0] = advection_velocity_image[2*k] + (advection_velocity_image[2*k] - advection_velocity_image[2*(k-ny)]) / 2.0;
            v[i][j][0][1] = advection_velocity_image[2*k+1] + (advection_velocity_image[2*k+1] - advection_velocity_image[2*(k-ny)+1]) / 2.0;
        } else {
            principal_angle = (principal_angle_image[k] + principal_angle_image[k+ny]) / 2.0;
            K1 = (diffusion_coefficient_image[k] + diffusion_coefficient_image[k+ny]) / 2.0;
            v[i][j][0][0] = (advection_velocity_image[2*k] + advection_velocity_image[2*(k+ny)]) / 2.0;
            v[i][j][0][1] = (advection_velocity_image[2*k+1] + advection_velocity_image[2*(k+ny)+1]) / 2.0;
        }
        principal_axis_from_angle(&e1x,&e1y,&e2x,&e2y,principal_angle);
        K2 = PARAM_RAT*K1;
        F_coeff_gradx[i][j][0] = K1*e1x*e1x + K2*e2x*e2x ;
        F_coeff_grady[i][j][0] = K1*e1x*e1y + K2*e2x*e2y ;

        if (j == ny-1) {
            principal_angle = principal_angle_image[k] + (principal_angle_image[k] - principal_angle_image[k-1]) / 2.0;
            K1 = diffusion_coefficient_image[k] + (diffusion_coefficient_image[k] - diffusion_coefficient_image[k-1]) / 2.0;
            v[i][j][1][0] = advection_velocity_image[2*k] + (advection_velocity_image[2*k] - advection_velocity_image[2*k-2]) / 2.0;
            v[i][j][1][1] = advection_velocity_image[2*k+1] + (advection_velocity_image[2*k+1] - advection_velocity_image[2*k-2+1]) / 2.0;
        } else {
            principal_angle = (principal_angle_image[k] + principal_angle_image[k+1]) / 2.0;
            K1 = (diffusion_coefficient_image[k] + diffusion_coefficient_image[k+1]) / 2.0;
            v[i][j][1][0] = (advection_velocity_image[2*k] + advection_velocity_image[2*k+2]) / 2.0;
            v[i][j][1][1] = (advection_velocity_image[2*k+1] + advection_velocity_image[2*k+2+1]) / 2.0;
        }
        principal_axis_from_angle(&e1x,&e1y,&e2x,&e2y,principal_angle);
        K2 = PARAM_RAT*K1;
        F_coeff_gradx[i][j][1] = K1*e1y*e1x + K2*e2y*e2x ;
        F_coeff_grady[i][j][1] = K1*e1y*e1y + K2*e2y*e2y ;


        if (i == 0) {
            principal_angle = principal_angle_image[k] - (principal_angle_image[k] - principal_angle_image[k+ny]) / 2.0;
            K1 = diffusion_coefficient_image[k] + (diffusion_coefficient_image[k] - diffusion_coefficient_image[k+ny]) / 2.0;
            v[i][j][2][0] = advection_velocity_image[2*k] + (advection_velocity_image[2*k] - advection_velocity_image[2*(k+ny)]) / 2.0;
            v[i][j][2][1] = advection_velocity_image[2*k+1] + (advection_velocity_image[2*k+1] - advection_velocity_image[2*(k+ny)+1]) / 2.0;
        } else {
            principal_angle = (principal_angle_image[k] + principal_angle_image[k-ny]) / 2.0;
            K1 = (diffusion_coefficient_image[k] + diffusion_coefficient_image[k-ny]) / 2.0;
            v[i][j][2][0] = (advection_velocity_image[2*k] + advection_velocity_image[2*(k-ny)]) / 2.0;
            v[i][j][2][1] = (advection_velocity_image[2*k+1] + advection_velocity_image[2*(k-ny)+1]) / 2.0;
        }
        principal_axis_from_angle(&e1x,&e1y,&e2x,&e2y,principal_angle);
        K2 = PARAM_RAT*K1;
        F_coeff_gradx[i][j][2] = K1*e1x*e1x + K2*e2x*e2x ;
        F_coeff_grady[i][j][2] = K1*e1x*e1y + K2*e2x*e2y ;

        if (j == 0) {
            principal_angle = principal_angle_image[k] - (principal_angle_image[k] - principal_angle_image[k+1]) / 2.0;
            K1 = diffusion_coefficient_image[k] + (diffusion_coefficient_image[k] - diffusion_coefficient_image[k+1]) / 2.0;
            v[i][j][3][0] = advection_velocity_image[2*k] + (advection_velocity_image[2*k] - advection_velocity_image[2*k+2]) / 2.0;
            v[i][j][3][1] = advection_velocity_image[2*k+1] + (advection_velocity_image[2*k+1] - advection_velocity_image[2*k+2+1]) / 2.0;
        } else {
            principal_angle = (principal_angle_image[k] + principal_angle_image[k-1]) / 2.0;
            K1 = (diffusion_coefficient_image[k] + diffusion_coefficient_image[k-1]) / 2.0;
            v[i][j][3][0] = (advection_velocity_image[2*k] + advection_velocity_image[2*k-2]) / 2.0;
            v[i][j][3][1] = (advection_velocity_image[2*k+1] + advection_velocity_image[2*k-2+1]) / 2.0;
        }
        principal_axis_from_angle(&e1x,&e1y,&e2x,&e2y,principal_angle);
        K2 = PARAM_RAT*K1;
        F_coeff_gradx[i][j][3] = K1*e1y*e1x + K2*e2y*e2x ;
        F_coeff_grady[i][j][3] = K1*e1y*e1y + K2*e2y*e2y ;

        T[i][j] = correlation_time_image[k] + SMALL;

        /* for timestep */
        double Ktot = K1+K2;
        if(Ktot > *Kmax) *Kmax = Ktot;
        double Vtot = fabs( v[i][j][0][0] ) + fabs( v[i][j][0][1] ) ;
        if(Vtot > *Vmax) *Vmax = Vtot;
        k++;
        //fprintf(stderr,"%d %d %g %g %g\n",i,j,*Kmax,K1,K2);

        /*
        fprintf(stderr,"%d %d %g %g %g %g\n",i,j,
            F_coeff_gradx[i][j][0],
            F_coeff_grady[i][j][0],
            F_coeff_gradx[i][j][1],
            F_coeff_grady[i][j][1]);
        */
    }
}


void get_diffusion_tensor_image(
    int nx,
    int ny,
    double* F_coeff_grad,
    double PARAM_RAT,
    double* principal_angle_image,
    double* diffusion_coefficient_image
    ) {
    void principal_axis_from_angle(double *e1x, double *e1y, double *e2x, double *e2y, double principal_angle);

    /* preparatory work: calculate some grid functions */
    int i,j,k,m;
    double e1x,e1y,e2x,e2y;
    double K1,K2;
    double principal_angle;

    k=0;
    m=0;
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {

        if (i == nx-1) {
            principal_angle = principal_angle_image[k] + (principal_angle_image[k] - principal_angle_image[k-ny]) / 2.0;
            K1 = diffusion_coefficient_image[k] + (diffusion_coefficient_image[k] - diffusion_coefficient_image[k-ny]) / 2.0;
        } else {
            principal_angle = (principal_angle_image[k] + principal_angle_image[k+ny]) / 2.0;
            K1 = (diffusion_coefficient_image[k] + diffusion_coefficient_image[k+ny]) / 2.0;
        }
        principal_axis_from_angle(&e1x,&e1y,&e2x,&e2y,principal_angle);
        K2 = PARAM_RAT*K1;
        F_coeff_grad[m++] = K1*e1x*e1x + K2*e2x*e2x ;
        F_coeff_grad[m++] = K1*e1x*e1y + K2*e2x*e2y ;

        if (j == ny-1) {
            principal_angle = principal_angle_image[k] + (principal_angle_image[k] - principal_angle_image[k-1]) / 2.0;
            K1 = diffusion_coefficient_image[k] + (diffusion_coefficient_image[k] - diffusion_coefficient_image[k-1]) / 2.0;
        } else {
            principal_angle = (principal_angle_image[k] + principal_angle_image[k+1]) / 2.0;
            K1 = (diffusion_coefficient_image[k] + diffusion_coefficient_image[k+1]) / 2.0;
        }
        principal_axis_from_angle(&e1x,&e1y,&e2x,&e2y,principal_angle);
        K2 = PARAM_RAT*K1;
        F_coeff_grad[m++] = K1*e1y*e1x + K2*e2y*e2x ;
        F_coeff_grad[m++] = K1*e1y*e1y + K2*e2y*e2y ;


        if (i == 0) {
            principal_angle = principal_angle_image[k] - (principal_angle_image[k] - principal_angle_image[k+ny]) / 2.0;
            K1 = diffusion_coefficient_image[k] + (diffusion_coefficient_image[k] - diffusion_coefficient_image[k+ny]) / 2.0;
        } else {
            principal_angle = (principal_angle_image[k] + principal_angle_image[k-ny]) / 2.0;
            K1 = (diffusion_coefficient_image[k] + diffusion_coefficient_image[k-ny]) / 2.0;
        }
        principal_axis_from_angle(&e1x,&e1y,&e2x,&e2y,principal_angle);
        K2 = PARAM_RAT*K1;
        F_coeff_grad[m++] = K1*e1x*e1x + K2*e2x*e2x ;
        F_coeff_grad[m++] = K1*e1x*e1y + K2*e2x*e2y ;

        if (j == 0) {
            principal_angle = principal_angle_image[k] - (principal_angle_image[k] - principal_angle_image[k+1]) / 2.0;
            K1 = diffusion_coefficient_image[k] + (diffusion_coefficient_image[k] - diffusion_coefficient_image[k+1]) / 2.0;
        } else {
            principal_angle = (principal_angle_image[k] + principal_angle_image[k-1]) / 2.0;
            K1 = (diffusion_coefficient_image[k] + diffusion_coefficient_image[k-1]) / 2.0;
        }
        principal_axis_from_angle(&e1x,&e1y,&e2x,&e2y,principal_angle);
        K2 = PARAM_RAT*K1;
        F_coeff_grad[m++] = K1*e1y*e1x + K2*e2y*e2x ;
        F_coeff_grad[m++] = K1*e1y*e1y + K2*e2y*e2y ;

        k++;
    }
}


void principal_axis_from_angle(double *e1x, double *e1y, double *e2x, double *e2y, double principal_angle)
{
    double s,c;
    c = cos(principal_angle);
    s = sin(principal_angle);
    *e1x = c;
    *e1y = s;
    *e2x = -s;
    *e2y = c;
}

void get_laplacian_image(
    int nt, int nx, int ny,
    double* lap,
    double PARAM_RAT,
    double* principal_angle_image,
    double* diffusion_coefficient_image,
    double* advection_velocity_image,
    double* correlation_time_image,
    double* del)
{
    void grid_function_calc(int nx, int ny, double F_coeff_gradx[nx][ny][4], double F_coeff_grady[nx][ny][4],
        double v[nx][ny][4][2], double T[nx][ny], double *Kmax, double *Vmax,
        double PARAM_RAT, double* principal_angle_image, double* advection_velocity_image,
        double* diffusion_coefficient_image, double* correlation_time_image);

    double gradx,grady,Fxp,Fxm,Fyp,Fym;
    int i,j,ip,jp,im,jm,n;

    /* calculate some grid functions */
    double T[nx][ny];
    double v[nx][ny][4][2];
    double F_coeff_gradx[nx][ny][4];
    double F_coeff_grady[nx][ny][4];
    memset(F_coeff_gradx, 0, sizeof(double) * nx * ny);
    memset(F_coeff_grady, 0, sizeof(double) * nx * ny);

    double Kmax = 0.;
    double Vmax = 0.;

    double dx = 1.0/nx;
    double dy = 1.0/ny;

    grid_function_calc(nx, ny, F_coeff_gradx, F_coeff_grady, v, T, &Kmax, &Vmax,
        PARAM_RAT, principal_angle_image, advection_velocity_image,
        diffusion_coefficient_image, correlation_time_image);

    double d = fmin(dx,dy);
    double cour = 0.45;
    double dtdiff = cour*0.25*d*d/Kmax;
    double dtadv = cour*0.5*d/Vmax;
    double dt = fmin(dtdiff, dtadv);


    for(n=0;n<nt;n++) {
        for(i=0;i<nx;i++)
        for(j=0;j<ny;j++) {
            ip = (i+nx+1)%nx ;
            im = (i+nx-1)%nx ;
            jp = (j+ny+1)%ny ;
            jm = (j+ny-1)%ny ;

            /* F = -K1 e1 (e1 . grad) - K2 e2 (e2 . grad) */
            /* gradient, centered at ...  */
            /* upper x face */
            gradx = (del[n*nx*ny+ip*nx+j] - del[n*nx*ny+i*nx+j])/dx;
            grady = 0.5*(
                (del[n*nx*ny+i*nx+jp] - del[n*nx*ny+i*nx+jm])/(2.*dy) +
                (del[n*nx*ny+ip*nx+jp] - del[n*nx*ny+ip*nx+jm])/(2.*dy)
                );
            Fxp = -(
                F_coeff_gradx[i][j][0]*gradx +
                F_coeff_grady[i][j][0]*grady
                );

            /* upper y face */
            gradx = 0.5*(
                (del[n*nx*ny+ip*nx+j] - del[n*nx*ny+im*nx+j])/(2.*dx) +
                (del[n*nx*ny+ip*nx+jp] - del[n*nx*ny+im*nx+jp])/(2.*dx)
                );
            grady = (del[n*nx*ny+i*nx+jp] - del[n*nx*ny+i*nx+j])/dy;
            Fyp = -(
                F_coeff_gradx[i][j][1]*gradx +
                F_coeff_grady[i][j][1]*grady
                );

            /* lower x face */
            gradx = (del[n*nx*ny+i*nx+j] - del[n*nx*ny+im*nx+j])/dx;
            grady = 0.5*(
                (del[n*nx*ny+i*nx+jp] - del[n*nx*ny+i*nx+jm])/(2.*dy) +
                (del[n*nx*ny+im*nx+jp] - del[n*nx*ny+im*nx+jm])/(2.*dy)
                );
            Fxm = -(
                F_coeff_gradx[i][j][2]*gradx +
                F_coeff_grady[i][j][2]*grady
                );

            /* lower y face */
            gradx = 0.5*(
                (del[n*nx*ny+ip*nx+j] - del[n*nx*ny+im*nx+j])/(2.*dx) +
                (del[n*nx*ny+ip*nx+jm] - del[n*nx*ny+im*nx+jm])/(2.*dx)
                );
            grady = (del[n*nx*ny+i*nx+j] - del[n*nx*ny+i*nx+jm])/dy;
            Fym = -(
                F_coeff_gradx[i][j][3]*gradx +
                F_coeff_grady[i][j][3]*grady
                );

            lap[n*nx*ny + i*nx + j] = -(Fxp - Fxm)/dx - (Fyp - Fym)/dy ;
        }
    }
}



void evolve_diffusion(int nx, int ny, double del[nx][ny], double F_coeff_gradx[nx][ny][4], double F_coeff_grady[nx][ny][4],
    double dt)
{

    double ddel[nx][ny];
    double gradx,grady,Fxp,Fxm,Fyp,Fym,deldiff;
    double dx=1.0/nx;
    double dy=1.0/ny;
    int i,j,ip,jp,im,jm;


#pragma omp parallel \
  shared ( del, ddel, dx, dy, F_coeff_gradx, F_coeff_grady ) \
  private (i, j, ip, im, jp, jm, gradx, grady, \
           Fxp, Fxm, Fyp, Fym, deldiff)
{
#pragma omp for

    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        ip = (i+nx+1)%nx ;
        im = (i+nx-1)%nx ;
        jp = (j+ny+1)%ny ;
        jm = (j+ny-1)%ny ;

        /* F = -K1 e1 (e1 . grad) - K2 e2 (e2 . grad) */
        /* gradient, centered at ...  */
        /* upper x face */
        gradx = (del[ip][j] - del[i][j])/dx;
        grady = 0.5*(
            (del[i][jp] - del[i][jm])/(2.*dy) +
            (del[ip][jp] - del[ip][jm])/(2.*dy)
            );
        Fxp = -(
            F_coeff_gradx[i][j][0]*gradx +
            F_coeff_grady[i][j][0]*grady
            );

        /* upper y face */
        gradx = 0.5*(
            (del[ip][j] - del[im][j])/(2.*dx) +
            (del[ip][jp] - del[im][jp])/(2.*dx)
            );
        grady = (del[i][jp] - del[i][j])/dy;
        Fyp = -(
            F_coeff_gradx[i][j][1]*gradx +
            F_coeff_grady[i][j][1]*grady
            );

        /* lower x face */
        gradx = (del[i][j] - del[im][j])/dx;
        grady = 0.5*(
            (del[i][jp] - del[i][jm])/(2.*dy) +
            (del[im][jp] - del[im][jm])/(2.*dy)
            );
        Fxm = -(
            F_coeff_gradx[i][j][2]*gradx +
            F_coeff_grady[i][j][2]*grady
            );

        /* lower y face */
        gradx = 0.5*(
            (del[ip][j] - del[im][j])/(2.*dx) +
            (del[ip][jm] - del[im][jm])/(2.*dx)
            );
        grady = (del[i][j] - del[i][jm])/dy;
        Fym = -(
            F_coeff_gradx[i][j][3]*gradx +
            F_coeff_grady[i][j][3]*grady
            );

        deldiff = -(Fxp - Fxm)/dx - (Fyp - Fym)/dy;

        ddel[i][j] = deldiff ;


    }
}

#pragma omp parallel \
  shared ( del, ddel, dt ) \
  private (i, j )
{
#pragma omp for
    /* update del */
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        del[i][j] += dt*ddel[i][j] ;
    }

}

}

void linear_mc(double x1, double x2, double x3, double *lout, double *rout)
{
    double Dqm,Dqp,Dqc,s;

    Dqm = 2. * (x2 - x1);
    Dqp = 2. * (x3 - x2);
    Dqc = 0.5 * (x3 - x1);

    s = Dqm * Dqp;

    if (s <= 0.)
    s = 0.;
    else {
    	if (fabs(Dqm) < fabs(Dqp) && fabs(Dqm) < fabs(Dqc))
        	s = Dqm;
        else if (fabs(Dqp) < fabs(Dqc))
            s = Dqp;
        else
            s = Dqc;
    }

    /* reconstruct left, right */
    *lout = x2 - 0.5*s;
    *rout = x2 + 0.5*s;
}

void reconstruct_lr(double d0, double d1, double d2, double d3, double *d_left, double *d_right)
{
	void linear_mc(double x1, double x2, double x3, double *lout, double *rout);
    double lout,rout;

	linear_mc(d0,d1,d2,&lout,&rout);
	*d_left = rout ;
	linear_mc(d1,d2,d3,&lout,&rout);
	*d_right = lout ;
}

double lr_to_flux(double d_left, double d_right, double v)
{
	double F;
	F = 0.5*(d_left*v + d_right*v) - fabs(v)*(d_right - d_left) ;
	return(F);
}

void evolve_advection(int nx, int ny, double del[nx][ny], double v[nx][ny][4][2], double dt)
{

    double ddel[nx][ny],Fxp,Fyp,Fxm,Fym,deladv;
    int i,j,im,jm,ip,jp,imm,jmm;
    double Fx[nx][ny];
    double Fy[nx][ny];
    double delr, dell;

    double dx = 1.0/nx;
    double dy = 1.0/ny;

#pragma omp parallel \
  shared ( del, v, Fx, Fy ) \
  private (i, j, imm, jmm, im, jm, ip, jp, dell, delr )
{
#pragma omp for
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        ip = (i+nx+1)%nx ;
        im = (i+nx-1)%nx ;
        imm = (i+nx-2)%nx ;
        jp = (j+ny+1)%ny ;
        jm = (j+ny-1)%ny ;
        jmm = (j+ny-2)%ny ;

        reconstruct_lr(del[imm][j],del[im][j],del[i][j],del[ip][j], &dell, &delr);
        Fx[i][j] = lr_to_flux(dell, delr, v[i][j][2][0]);

        reconstruct_lr(del[i][jmm],del[i][jm],del[i][j],del[i][jp], &dell, &delr);
        Fy[i][j] = lr_to_flux(dell, delr, v[i][j][3][1]);
    }

}
#pragma omp parallel \
  shared ( ddel, Fx, Fy ) \
  private (i, j, ip, jp, Fxp, Fyp, Fxm, Fym, deladv )
{
#pragma omp for
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        ip = (i+nx+1)%nx ;
        jp = (j+ny+1)%ny ;

		Fxp = Fx[ip][j];
		Fyp = Fy[i][jp];
		Fxm = Fx[i][j];
		Fym = Fy[i][j];

		deladv = -(Fxp - Fxm)/dx - (Fyp - Fym)/dy;

		ddel[i][j] = deladv ;
	}
}

#pragma omp parallel \
  shared ( del, ddel, dt ) \
  private (i, j )
{
#pragma omp for
    /* update del */
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        del[i][j] += dt*ddel[i][j] ;
    }

}

}

void evolve_noise(int nx, int ny, double del[nx][ny], double dt, double PARAM_EPS, gsl_rng* r)
{
    int i,j;
	double del_noise[nx][ny];
    void noise_model(int nx, int ny, double del_noise[nx][ny], double dt, double PARAM_EPS, gsl_rng *r);

    noise_model(nx, ny, del_noise, dt, PARAM_EPS, r);

    /* update del */
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        del[i][j] += del_noise[i][j];
    }

}

void evolve_source(int nx, int ny, double del[nx][ny], double dt, double* source)
{
    int i,j;
    int n = 0;
    /* update del */
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        del[i][j] += dt * source[n];
        n++;
    }

}

void evolve_decay(int nx, int ny, double del[nx][ny], double T[nx][ny], double dt)
{
	int i,j;
    double Tdec;

#pragma omp parallel \
  shared ( del, dt ) \
  private (i, j, Tdec )
{
#pragma omp for
    /* update del */
    for(i=0;i<nx;i++)
    for(j=0;j<ny;j++) {
        Tdec = T[i][j] + 2.*dt ;
        //del[i][j] *= (1. - 0.5*dt/Tdec)/(1. + 0.5*dt/Tdec) ;
        del[i][j] += -dt*del[i][j]/Tdec;
    }
}

}
