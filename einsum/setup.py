#setup.py
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='intg_cython',
    ext_modules=cythonize("test.pyx"),
    zip_safe=False,
)