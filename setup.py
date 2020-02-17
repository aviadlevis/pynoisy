from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        'pynoisy', [ 'src/noisy.pyx'],
        include_dirs=['src/'],
        libraries=['gsl', 'blas'],
        library_dirs=['/usr/lib/x86_64-linux-gnu/'],
        depends=['src/noisy.h', 'src/evolve.c', 'src/image.c', 'src/model_disk.c', 'src/model_uniform.c']
    )
]

setup(
    name='pynoisy',
    ext_modules=cythonize(extensions)
)
