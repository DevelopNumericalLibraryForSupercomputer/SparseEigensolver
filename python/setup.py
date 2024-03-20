# setup.py

from setuptools import setup
from Cython.Build import cythonize

from setuptools.extension import Extension
import numpy

extensions = [
    Extension('PyIterativeSolver', ["PyIterativeSolver.pyx"],
              libraries=['mkl_rt'],
              library_dirs=['/opt/intel/oneapi/mkl/2024.0/lib'],
              include_dirs=['/opt/intel/oneapi/mkl/2024.0/include'],)
]
setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()]
)