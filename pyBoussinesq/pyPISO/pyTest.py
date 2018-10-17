import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, inv
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



import fvUtilities as fv


n_nodes = 5
n_in_nodes = (n_nodes-2)**2
x_axis = np.linspace(0., 1., num = n_nodes)
y_axis = x_axis[:]

x,y = np.meshgrid(x_axis, y_axis)      # f[i_y,j_x]

fv.init_mesh((x,y))
U = np.zeros((2,n_nodes,n_nodes))
#a = fv.in_matrix_coef('div(phi, .)', (fv.flux(U)))
#b = fv.in_matrix_coef('div(phi, .)', (fv.flux(U)))

in_row, in_col =  fv.internal_rows_columns()

A = csr_matrix((fv.in_matrix_coef('dV',()).reshape(n_in_nodes*5), (in_row, in_col)), 
		shape=(n_nodes*n_nodes,n_nodes*n_nodes), 
		dtype = np.float64)
B = csr_matrix((fv.in_matrix_coef('div(phi, .)', (fv.flux(U))).reshape(n_in_nodes*5), (in_row, in_col)), 
		shape=(n_nodes*n_nodes,n_nodes*n_nodes), 
		dtype = np.float64)

print A+B

