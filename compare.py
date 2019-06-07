import numpy as np
import tensorly
from tensorly import decomposition
from scipy.io import loadmat
import sys
try:
    r = int(sys.argv[1])
except:
    r = 1

tensor_res = loadmat('tensor_0_7.mat')['t_new']
print(np.shape(tensor_res))
a, b = decomposition.parafac(tensor=tensor_res, rank=r, verbose=2, return_errors=1)