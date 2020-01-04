# noisy

Noisy solves an advection-diffusion-decay equation that is forced by a 
noise model.  Generates a gaussian random field.

Model parameters are set in ```noisy.h```.

Output is a series of ppm files in subdirectory ```images```, and an
ascii snapshot of the final state in ```noisy.out```.

Two models are currently available.  Choose a model by editing 
```makefile```.  

```model_uniform.c``` has constant advection velocity, set to 0 by default.
    anisotropic diffusion, unit decay time, and a constant-in-time 
    white-noise forcing.  This converges quickly to a steady-state 
    solution of the diffusion-decay equation.  This model is useful
    for testing the code.
  
```model_disk.c``` mocks up an accretion disk on the sky.  Diffusion 
    coefficients decay time, and orientation of the diffusion tensor 
    are all space dependent.  Forcing is white noise in space *and* time.

Follows Lee & Gammie 2020, based on
[Lindgren, Rue, and Lindstr\:om 2011, J.R. Statist. Soc. B 73, pp 423-498]
(https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2011.00777.x)

In particular, noisy implements eq. 17, which has power spectrum given by 
eq. 18.

4 Jan 2020


