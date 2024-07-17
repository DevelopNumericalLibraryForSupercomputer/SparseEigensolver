# HOWTO build: python setup.py build_ext --inplace

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Intel 컴파일러 설정
os.environ["CC"] = "/opt/intel/oneapi/mpi/2021.11/bin/mpiicpx"
os.environ["CXX"] = "/opt/intel/oneapi/mpi/2021.11/bin/mpiicpx"

extensions = [
    Extension(
        "PySparseTensor",  # Module name
        sources=["pydensetensor.pyx"],
        libraries=['mkl_rt', 'm'],
        include_dirs=[
            np.get_include(),  # numpy header file path
            "../include"  # C++ source file path (parent folder)
        ],
        language="c++",  
        extra_compile_args=["-std=c++17"],  # C++17
    )
]

setup(
    name="PySparseTensor",
    version="0.1",
    ext_modules=cythonize(extensions, language_level="3", annotate=True), # language_level="3" : python3, annotate=True : write html file
    zip_safe=False,
)