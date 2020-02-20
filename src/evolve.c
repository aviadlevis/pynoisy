
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
    double F_coeff_gradx[N][N][4], 
    double F_coeff_grady[N][N][4], 
    double v[N][N][4][2],
    double T[N][N],
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
    for(i=0;i<N;i++) 
    for(j=0;j<N;j++) {

        if (i == N-1) {
            principal_angle = principal_angle_image[k] + (principal_angle_image[k] - principal_angle_image[k-N]) / 2.0;
            K1 = diffusion_coefficient_image[k] + (diffusion_coefficient_image[k] - diffusion_coefficient_image[k-N]) / 2.0;
            v[i][j][0][0] = advection_velocity_image[2*k] + (advection_velocity_image[2*k] - advection_velocity_image[2*(k-N)]) / 2.0;
            v[i][j][0][1] = advection_velocity_image[2*k+1] + (advection_velocity_image[2*k+1] - advection_velocity_image[2*(k-N)+1]) / 2.0;
        } else {
            principal_angle = (principal_angle_image[k] + principal_angle_image[k+N]) / 2.0;
            K1 = (diffusion_coefficient_image[k] + diffusion_coefficient_image[k+N]) / 2.0;
            v[i][j][0][0] = (advection_velocity_image[2*k] + advection_velocity_image[2*(k+N)]) / 2.0;
            v[i][j][0][1] = (advection_velocity_image[2*k+1] + advection_velocity_image[2*(k+N)+1]) / 2.0;
        }
        principal_axis_from_angle(&e1x,&e1y,&e2x,&e2y,principal_angle);
        K2 = PARAM_RAT*K1;
        F_coeff_gradx[i][j][0] = K1*e1x*e1x + K2*e2x*e2x ;
        F_coeff_grady[i][j][0] = K1*e1x*e1y + K2*e2x*e2y ;

        if (j == N-1) {
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
            principal_angle = principal_angle_image[k] - (principal_angle_image[k] - principal_angle_image[k+N]) / 2.0;
            K1 = diffusion_coefficient_image[k] + (diffusion_coefficient_image[k] - diffusion_coefficient_image[k+N]) / 2.0;
            v[i][j][2][0] = advection_velocity_image[2*k] + (advection_velocity_image[2*k] - advection_velocity_image[2*(k+N)]) / 2.0;
            v[i][j][2][1] = advection_velocity_image[2*k+1] + (advection_velocity_image[2*k+1] - advection_velocity_image[2*(k+N)+1]) / 2.0;
        } else {
            principal_angle = (principal_angle_image[k] + principal_angle_image[k-N]) / 2.0;
            K1 = (diffusion_coefficient_image[k] + diffusion_coefficient_image[k-N]) / 2.0;
            v[i][j][2][0] = (advection_velocity_image[2*k] + advection_velocity_image[2*(k-N)]) / 2.0;
            v[i][j][2][1] = (advection_velocity_image[2*k+1] + advection_velocity_image[2*(k-N)+1]) / 2.0;
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

void evolve_diffusion(double del[N][N], double F_coeff_gradx[N][N][4], double F_coeff_grady[N][N][4],
    double dt)
{

    double ddel[N][N];
    double gradx,grady,Fxp,Fxm,Fyp,Fym,deldiff;
    double dx=PARAM_FOV/N;
    double dy=PARAM_FOV/N;
    int i,j,ip,jp,im,jm;


#pragma omp parallel \
  shared ( del, ddel, dx, dy, F_coeff_gradx, F_coeff_grady ) \
  private (i, j, ip, im, jp, jm, gradx, grady, \
           Fxp, Fxm, Fyp, Fym, deldiff)
{
#pragma omp for

    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        ip = (i+N+1)%N ;
        im = (i+N-1)%N ;
        jp = (j+N+1)%N ;
        jm = (j+N-1)%N ;

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
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
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

void evolve_advection(double del[N][N], double v[N][N][4][2], double dt)
{

    double ddel[N][N],Fxp,Fyp,Fxm,Fym,deladv;
    int i,j,im,jm,ip,jp,imm,jmm;
    double Fx[N][N];
    double Fy[N][N];
    double delr, dell;

    double dx = PARAM_FOV/N;
    double dy = PARAM_FOV/N;

#pragma omp parallel \
  shared ( del, v, Fx, Fy ) \
  private (i, j, imm, jmm, im, jm, ip, jp, dell, delr )
{
#pragma omp for
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        ip = (i+N+1)%N ;
        im = (i+N-1)%N ;
        imm = (i+N-2)%N ;
        jp = (j+N+1)%N ;
        jm = (j+N-1)%N ;
        jmm = (j+N-2)%N ;

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
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        ip = (i+N+1)%N ;
        jp = (j+N+1)%N ;

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
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        del[i][j] += dt*ddel[i][j] ;
    }

}

}

void evolve_noise(double del[N][N], double dt, double PARAM_EPS)
{
    int i,j;
	double del_noise[N][N];
    void noise_model(double del_noise[N][N], double dt, double PARAM_EPS);

    noise_model(del_noise, dt, PARAM_EPS);

    /* update del */
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        del[i][j] += del_noise[i][j];
    }

}

void evolve_decay(double del[N][N], double T[N][N], double dt)
{
	int i,j;
    double Tdec;

#pragma omp parallel \
  shared ( del, dt ) \
  private (i, j, Tdec )
{
#pragma omp for
    /* update del */
    for(i=0;i<N;i++)
    for(j=0;j<N;j++) {
        Tdec = T[i][j] + 2.*dt ;
        //del[i][j] *= (1. - 0.5*dt/Tdec)/(1. + 0.5*dt/Tdec) ;
        del[i][j] += -dt*del[i][j]/Tdec;
    }
}

}
