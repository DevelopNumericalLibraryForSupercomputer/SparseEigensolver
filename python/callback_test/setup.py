from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "matvec",
        ["matvec.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11", "-fPIC"],
        extra_link_args=["-shared"]
    )
]

setup(
    name="matvec",
    ext_modules=cythonize(ext_modules, language_level="3"),
    include_dirs=[np.get_include()],
)