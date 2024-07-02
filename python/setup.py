from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Intel 컴파일러 설정
os.environ["CC"] = "/opt/intel/oneapi/mpi/2021.11/bin/mpiicpx"
os.environ["CXX"] = "/opt/intel/oneapi/mpi/2021.11/bin/mpiicpx"

# C++ 소스 파일 리스트
#cpp_sources = ['Decomposer.cpp']  # C++ 구현 파일

# Cython 확장 모듈 정의
#pycomm.pyx 파일을 컴파일
extensions = [
    Extension(
        "PySparseTensor",  # 생성될 Python 모듈 이름
        sources=["pycomm.pyx"],
        libraries=['mkl_rt', 'm'],
        include_dirs=[
            np.get_include(),  # numpy 헤더 파일 경로
            "../include"  # C++ 소스 파일 경로
        ],
        language="c++",  # C++을 사용한다고 명시
        extra_compile_args=["-std=c++17"],  # C++11 표준 사용 (필요에 따라 조정)
    )
]

setup(
    name="PySparseTensor",
    version="0.1",
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
)