from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        'pynoisy', ['src/noisy.pyx'],
        #extra_compile_args = ['-Wunknown-pragmas', '-Wreturn-type'],
        include_dirs=['src/', numpy.get_include()],
        libraries=['gsl', 'blas'],
        library_dirs=['/usr/lib/x86_64-linux-gnu/'],
        depends=['src/noisy.h', 'src/evolve.c', 'src/image.c', 'src/model_disk.c', 'src/model_uniform.c']
    )
]

setup(
    name='pynoisy',
    include_dirs=[numpy.get_include()],
    ext_modules=cythonize(extensions)
)
