import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.sparse import csc_array
from scipy.sparse.linalg import spsolve_triangular

J1 = ([[1, 0.5, 0.7], 
        [1, 4, 3],
        [1, 0.45, 3]])

F1 = np.array([1, 1.5, 2], dtype=float)

J = csc_array([[1, 1/8, 0, 1/4, 0, 0, 0, 0, 0,], 
                [3/8, 1, 1/8, 0, 1/4, 0, 0, 0, 0], 
                [0, 3/8, 9/8, 0, 0, 1/4, 0, 0 , 0], 
                [1/4, 0, 0, 1, 1/8, 0, 1/4, 0 , 0], 
                [0, 1/4, 0, 3/8, 1, 1/8, 0, 1/4, 0],
                [0, 0, 1/4, 0, 3/8, 9/8, 0, 0 , 1/4], 
                [0, 0, 0, 1/4, 0, 0, 1, 1/8 , 0], 
                [0, 0, 0, 0, 1/4, 0, 3/8, 1 , 1/8], 
                [0, 0, 0, 0, 0, 1/4, 0, 3/8, 1]],
               dtype = float)
F = np.array([0.75,  0.75,  0.625, 1,    1,    0.875, 0.75,  0.75,  0.625], dtype = float)
H = spla.spsolve(J, F)
x = spsolve_triangular(J, -F)
#print(H)
print(np.min(J1))
    