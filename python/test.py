#import pycomm.pyx
from PySparseTensor import mkl_comm

print("Comm test")
print("Create comm")
new_comm = mkl_comm()
print("Create Comm using create_comm")
new_2comm = new_comm.clone()
