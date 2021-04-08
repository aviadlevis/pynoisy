pynoisy
---
pynoisy is:
1. A python wrapper for a modified version of the `inoisy` code [1] that supports arbitrary xarray matrices as diffusion tensor fields. 
pynoisy can be used to generate 3D (spatio-temporal) Gaussian Random Fields (GRFs) as solutions to a stochastic partial differential equation [2,3] (SPDE), which is solved using `HYPRE` computing library [4].   
`Tutorial1` within the tutorials directory gives a notebook example on generation of GRFs.

2. A tool for inferring parameters of stochastic fluid dynamics of black-hole accretion from Event Horizon Telescope (EHT) measurements. 
EHT measurements are Very Large Baseline Intereferometric (VLBI) measurements which are synthesized using `eht-imaging` [5]. 


Installation
----
Installation using using [anaconda](https://www.anaconda.com/) package management.  
The following installation steps assume that MPI (e.g. [openmpi](https://www.open-mpi.org/), [mpich](https://www.mpich.org/)) is installed and has been tested on Linux Ubuntu 18.04.5.

Clone pynoisy repository with the inoisy submodule
```
git clone --recurse-submodules https://github.com/aviadlevis/pynoisy
cd pynoisy
```
Clone and install [HYPRE](https://github.com/hypre-space/hypre) library. If HYPRE was previously installed make sure to have `HYPRE_DIR` point to the right path.
```
git clone https://github.com/hypre-space/hypre.git
cd hypre/src
./configure
make install
cd ../../
``` 
Start a virtual environment and export ./inoisy directory environment variable (this path will be used to mpirun compiled executables)
```
conda create -n pynoisy python=3.7.4
conda activate pynoisy
conda env config vars set INOISY_DIR=$(pwd)/inoisy HYPRE_DIR=$(pwd)/hypre/src/hypre LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HYPRE_DIR/lib/
conda activate pynoisy
```
Install [xarray](http://xarray.pydata.org/) and its dependencies and other required packages
```
conda install --file requirements.txt
conda install -c conda-forge xarray dask netCDF4 bottleneck
```





References
---
1. `inoisy`  [code](https://github.com/AFD-Illinois/inoisy)
2. Lee, D. and Gammie, C.F., 2021. [Disks as Inhomogeneous, Anisotropic Gaussian Random Fields](https://iopscience.iop.org/article/10.3847/1538-4357/abc8f3/meta).
   The Astrophysical Journal, 906(1), p.39.  
3.  Lindgren, F., Rue, H. and Lindström, J., 2011. [An explicit link between Gaussian fields and Gaussian Markov random 
fields: the stochastic partial differential equation approach](https://rss.onlinelibrary.wiley.com/doi/epdf/10.1111/j.1467-9868.2011.00777.x). Journal of the Royal Statistical Society: 
Series B (Statistical Methodology), 73(4), pp.423-498. 
4. `HYPRE` [computing library](https://github.com/hypre-space/hypre)
5.  `eht-imaging` [code](https://github.com/achael/eht-imaging)


##
This code was created by Aviad Levis, California Institute of Technology, 2020.

