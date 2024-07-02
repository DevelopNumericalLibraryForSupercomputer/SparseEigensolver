#import pycomm.pyx
import PySparseTensor

print("Comm test")
print("Create comm")
new_comm = PySparseTensor.create_mkl_comm()
print("Copy new_comm into new2comm")
new2_comm = PySparseTensor.copy_mkl_comm(new_comm)
