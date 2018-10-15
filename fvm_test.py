# FVM in a 2D rectangle domain
# Upgraded from https://github.com/danieljfarrell/FVM

from __future__ import division
import collections
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import dia_matrix
from scipy.sparse import csr_matrix
np.random.seed(seed = 1)

''' Checks that the index specified (can be number or an iterable) is strictly within the given range. '''
def check_index_within_bounds(i, i_min, i_max):
	success = np.all((i>=i_min)*(i<=i_max))
	if success:
		return True
	if isinstance(i, collections.Iterable):
		# The index is array-like
		print "Index is out of bounds. \ni=%s" % i[np.where(np.logical_not((i>=i_max)*(i<=i_min)))]
	else:
		# The index is an number
		print "Index is out of bounds. \ni=%s" % i
	return False

''' Class of 1D cell-centered axes defined by faces for the finite volume method '''
class Axis(object):
	def __init__(self, faces):
		super(Axis, self).__init__()

		# Check for duplicated points
		if len(faces) != len(set(faces)):
			raise ValueError("The face coordinate array contains duplicated positions. No cell can have zero volume.")
		self.faces = np.array(faces)
		self.cells = np.concatenate([ [faces[0]], 0.5 * (self.faces[:-1] + self.faces[1:]), [faces[-1]] ])
		# Two Ghost Cells added on the two sides.
		self.num = len(self.faces) - 1
		# self.num indicates the number of real cells, i.e. without counting the Ghost Cells.
		# self.cells has self.num+2 elements
		self.cell_widths = np.concatenate([[0.], (self.faces[1:] - self.faces[:-1]), [0.]])
		# Corresponding widths of cells. Ghost Cells have zero width.

	''' Width of the cell at the specified index. '''
	def h(self, i):
		return self.cell_widths[i]

	''' Distance between centroids in the backward direction. '''
	def hb(self, i):
		if not check_index_within_bounds(i, 1, self.num+1):
			raise ValueError("hb index runs out of bounds")
		return (self.cells[i] - self.cells[i-1])

	''' Distance between centroids in the forward direction. '''
	def hf(self, i):
		if not check_index_within_bounds(i, 0, self.num):
			raise ValueError("hf index runs out of bounds")
		return (self.cells[i+1] - self.cells[i])

''' Class of 2D mesh cell-centered mesh defined by two axes x and y '''
class Structured2DMesh(object):
	def __init__(self, x, y):
		super(Structured2DMesh, self).__init__()
		self.x = x
		self.y = y
		# Construction of cells
		# Cell numbers on x and on y
		Cx = len(x.cells)
		Cy = len(y.cells)
		# Face numbers on x and on y
		Fx = len(x.faces)
		Fy = len(y.faces)

		self.cells = np.arange(Cx * Cy).reshape(Cx, Cy)
		self.cell_j, self.cell_i = np.meshgrid(np.arange(Cy), np.arange(Cx))
		self.cell_i = self.cell_i.reshape(Cx * Cy)
		self.cell_j = self.cell_j.reshape(Cx * Cy)
		self.cell_x, self.cell_y = x.cells[self.cell_i], y.cells[self.cell_j]

		# Construction of faces
		self.faces = np.arange(Fx*Cy + Cx*Fy)
		## Faces to the direction of x, type of w and e
		self.faces_to_x = self.faces[:Fx*Cy].reshape(Fx, Cy)
		self.face_to_x_j, self.face_to_x_i = np.meshgrid(np.arange(Cy), np.arange(Fx))
		## Faces in direction of y, type of s and n
		self.faces_to_y = self.faces[Fx*Cy:].reshape(Cx, Fy)
		self.face_to_y_j, self.face_to_y_i = np.meshgrid(np.arange(Fy), np.arange(Cx))
		#
		self.face_x = np.concatenate([self.x.faces[self.face_to_x_i].reshape(Fx*Cy), self.x.cells[self.face_to_y_i].reshape(Cx*Fy)])
		self.face_y = np.concatenate([self.y.cells[self.face_to_x_j].reshape(Fx*Cy), self.y.faces[self.face_to_y_j].reshape(Cx*Fy)])
	# Define neighboring cells
	def W(self, P):
		ci, cj = self.cell_i[P], self.cell_j[P]
		return self.cells[np.where(ci-1>0, ci-1, 0), cj]
	def E(self, P):
		ci, cj = self.cell_i[P], self.cell_j[P]
		return self.cells[np.where(ci<=self.x.num, ci+1, self.x.num+1), cj]
	def S(self, P):
		ci, cj = self.cell_i[P], self.cell_j[P]
		return self.cells[ci, np.where(cj-1>0, cj-1, 0)]
	def N(self, P):
		ci, cj = self.cell_i[P], self.cell_j[P]
		return self.cells[ci, np.where(cj<=self.y.num, cj+1, self.y.num+1)]

	# Define neighboring faces

	# With Ghost Cells:
	# Cells: 0- -1- -2- -3
	# Faces:  -0- -1- -2-
	def w(self, P):
		ci, cj = self.cell_i[P], self.cell_j[P]
		return self.faces_to_x[np.where(ci-1>0, ci-1, 0), cj]
	def e(self, P):
		ci, cj = self.cell_i[P], self.cell_j[P]
		return self.faces_to_x[np.where(ci<=self.x.num, ci, self.x.num), cj]
	def s(self, P):
		ci, cj = self.cell_i[P], self.cell_j[P]
		return self.faces_to_y[ci, np.where(cj-1>0, cj-1, 0)]
	def n(self, P):
		ci, cj = self.cell_i[P], self.cell_j[P]
		return self.faces_to_y[ci, np.where(cj<=self.y.num, cj, self.y.num)]

	# Define i and j
	def i(self, P):
		return self.cell_i[P]
	def j(self, P):
		return self.cell_j[P]

	def show_mesh(self):
		import matplotlib.pyplot as plt
		P = self.cells
		iP, jP = self.i(P), self.j(P)
		plt.scatter(mesh.x.cells[iP], mesh.y.cells[jP], color = 'black', alpha = 0.25)
		P = self.cells[1:-1, 1:-1]
		iP, jP = self.i(P), self.j(P)
		plt.scatter(mesh.x.cells[iP], mesh.y.cells[jP], color = 'black', alpha = 1)
		#plt.xlim(self.x.faces[0], self.x.faces[-1])
		#plt.ylim(self.y.faces[0], self.y.faces[-1])
		x_min, x_max = self.x.faces[0], self.x.faces[-1]
		y_min, y_max = self.y.faces[0], self.y.faces[-1]
		for y_face in self.y.faces: plt.plot([x_min, x_max], [y_face]*2, color ='black', alpha = 0.5)
		for x_face in self.x.faces: plt.plot([x_face]*2, [y_min, y_max], color ='black', alpha = 0.5)
		plt.axis('equal')
		plt.show()



''' Representation of a variable defined at the cell centers. Provides interpolation functions to calculate the value at cell faces. '''
# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
class CellScalar(np.ndarray):
	def __new__(cls, input_array, mesh = None):
		# Conversion of 'input_array' to array if it is actually a function of (x,y)
		if hasattr(input_array, '__call__'):
			input_array = input_array(mesh.cell_x, mesh.cell_y)
		# Conversion of 'input_array' to array if it is actually just a constant
		else :
			try:
				len(input_array)
			except:
				input_array = input_array * np.ones(len(mesh.x.cells) * len(mesh.y.cells))

		obj = np.asarray(input_array).view(cls)
		obj.mesh = mesh
		return obj

	def __array_finalize__(self, obj):
		if obj is None: return
		self.mesh = getattr(obj, 'mesh', None)
		self.__get_items__ = getattr(obj, '__get_items__', None)

	# Interpolation needs to work with ndarray
	# Applicable for real cells only
	''' Linear interpolation of the cell value at the west face '''
	def w(self, P):
		W = self.mesh.W(P)
		iP, iW = self.mesh.i(P), self.mesh.i(W)
		x, y = self.mesh.x, self.mesh.y
		return (x.h(iP)*self[W] + x.h(iW)*self[P])/(2*x.hb(iP))
	''' Linear interpolation of the cell value at the east face '''
	def e(self, P):
		E = self.mesh.E(P)
		iP, iE = self.mesh.i(P), self.mesh.i(E)
		x, y = self.mesh.x, self.mesh.y
		return (x.h(iP)*self[E] + x.h(iE)*self[P])/(2*x.hf(iP))
	''' Linear interpolation of the cell value at the south face '''
	def s(self, P):
		S = self.mesh.S(P)
		jP, jS = self.mesh.j(P), self.mesh.j(S)
		x, y = self.mesh.x, self.mesh.y
		return (y.h(jP)*self[S] + y.h(jS)*self[P])/(2*y.hb(jP))
	''' Linear interpolation of the cell value at the north face '''
	def n(self, P):
		N = self.mesh.N(P)
		jP, jN = self.mesh.j(P), self.mesh.j(N)
		x, y = self.mesh.x, self.mesh.y
		return (y.h(jP)*self[N] + y.h(jN)*self[P])/(2*y.hf(jP))

	def on_faces(self):
		return FaceScalar(self)

''' Representation of a variable defined at the face centers. '''
class FaceScalar(np.ndarray):
	def __new__(cls, input_array, mesh = None):
		# Conversion of 'input_array' to array if it is actually a function of (x,y)
		if hasattr(input_array, '__call__'):
			input_array = input_array(mesh.face_x, mesh.face_y)
		# Conversion of 'input_array' to array if it is actually just a constant
		else :
			try:
				len(input_array)
			except:
				input_array = input_array * np.ones(len(mesh.x.faces)*len(mesh.y.cells) + len(mesh.x.cells)*len(mesh.y.faces))

		obj = np.asarray(input_array).view(cls)
		obj.mesh = mesh
		return obj

	def __array_finalize__(self, obj):
		if obj is None: return
		self.mesh = getattr(obj, 'mesh', None)
		self.__get_items__ = getattr(obj, '__get_items__', None)

class CellVector(np.ndarray):
	def __new__(cls, input_array, mesh = None):
		# Conversion of 'input_array' to array if it is actually a function of (x,y)
		if hasattr(input_array, '__call__'):
			input_array = input_array(mesh.cell_x, mesh.cell_y).T
		# Conversion of 'input_array' to array if it is actually just a constant
		else :
			try:
				len(input_array.T[0])
				# Length test on Vx
			except:
				Vx, Vy = input_array
				#input_array = input_array * np.ones(len(mesh.x.cells) * len(mesh.y.cells) * 2).reshape(len(mesh.x.cells) * len(mesh.y.cells), 2)
				#print '(Vx, Vy) = ', Vx, Vy
				input_array = np.array([Vx * np.ones(len(mesh.x.cells)*len(mesh.y.cells)),
				Vy * np.ones(len(mesh.x.cells)*len(mesh.y.cells))])
				return input_array.T

		obj = np.asarray(input_array).view(cls)
		obj.mesh = mesh
		return obj

	def __array_finalize__(self, obj):
		if obj is None: return
		self.mesh = getattr(obj, 'mesh', None)
		self.__get_items__ = getattr(obj, '__get_items__', None)

'''
Matrix for interior domain
'''
class fvMatrix(object):
	def __init__(self, operator, mesh=None):
		super(fvMatrix, self).__init__()
		self.mesh = mesh
		if(operator == 'lap'): self.laplacian()
	def laplacian(self):
		P = mesh.cells[1:-1,1:-1].reshape(mesh.x.num * mesh.y.num)
		self.row = np.concatenate([P]*5)
		self.col = np.concatenate([mesh.W(P), mesh.E(P), mesh.S(P), mesh.N(P), P])
		'''
		Demonstration descretization of laplacian operator
		W - w --- P --- e - E
		      h/2   h/2

		  d2T   dT |    dT |    TE - TP   TP - TW
		h --- = -- |  - -- |  = ------- - -------
		  dx2   dx |e   dx |w     hf         hb
		'''
		'''
		a={\
			'W': 1./mesh.x.hb(mesh.i(P))/mesh.x.h(mesh.i(P)),\
			'E': 1./mesh.x.hf(mesh.i(P))/mesh.x.h(mesh.i(P)),\
			'S': 1./mesh.y.hb(mesh.j(P))/mesh.y.h(mesh.j(P)),\
			'N': 1./mesh.y.hf(mesh.i(P))/mesh.y.h(mesh.j(P))\
		}
		'''
		a={\
			'W': 1./mesh.x.hb(mesh.i(P))*mesh.y.h(mesh.j(P)),\
			'E': 1./mesh.x.hf(mesh.i(P))*mesh.y.h(mesh.j(P)),\
			'S': 1./mesh.y.hb(mesh.j(P))*mesh.x.h(mesh.i(P)),\
			'N': 1./mesh.y.hf(mesh.j(P))*mesh.x.h(mesh.i(P))\
		}
		self.data = np.concatenate([a['W'], a['E'], a['S'], a['N'],\
			-a['W']-a['E']-a['S']-a['N']])
		#n_matrix = len(mesh.x.cells) * len(mesh.y.cells)
		#return csr_matrix((data, (row, col)), shape=(n_matrix, n_matrix), dtype = np.float64)
	def matrix(self):
		matrix_size = len(self.mesh.x.cells) * len(self.mesh.y.cells)
		return csr_matrix((self.data, (self.row, self.col)), shape = (matrix_size, matrix_size))

class fvBoundary(object):
	def __init__(self, mesh=None):
		super(fvBoundary, self).__init__()
		self.mesh = mesh
		self.row = np.array([])
		self.col = np.array([])
		self.data = np.array([])
		self.patch = {\
			'W': mesh.cells[ 0,:],\
			'E': mesh.cells[-1,:],\
			'S': mesh.cells[:, 0],\
			'N': mesh.cells[:,-1]\
		}
		self.patch_neighbor = {\
			'W': mesh.cells[ 1,:],\
			'E': mesh.cells[-2,:],\
			'S': mesh.cells[:, 1],\
			'N': mesh.cells[:,-2]\
		}
		self.B = np.zeros(len(mesh.x.cells)*len(mesh.y.cells))

	def add_constant_dirichlet(self, patch_name, constant):
		patch = [self.patch[patch_name][1:-1], self.patch_neighbor[patch_name][1:-1]]
		self.row = np.concatenate([self.row, patch[0], patch[0]])
		self.col = np.concatenate([self.col, patch[0], patch[1]])
		if(patch_name in ['W','E']): 
			h = [self.mesh.x.h(self.mesh.i(patch[0])), self.mesh.x.h(self.mesh.i(patch[1]))]
		if(patch_name in ['S','N']): 
			h = [self.mesh.y.h(self.mesh.j(patch[0])), self.mesh.y.h(self.mesh.j(patch[1]))]
		self.data = np.concatenate([self.data, h[1]/(h[0]+h[1]), h[0]/(h[0]+h[1])])
		self.B[patch[0]] += constant
	def add_constant_neumann(self, patch_name, constant):
		patch = [self.patch[patch_name], self.patch_neighbor[patch_name]]
		self.row = np.concatenate([self.row, patch[0], patch[0]])
		self.col = np.concatenate([self.col, patch[0], patch[1]])
		if(patch_name in ['W','E']): 
			h = self.mesh.x.cells[self.mesh.i(patch[0])] - self.mesh.x.cells[self.mesh.i(patch[1])]
		if(patch_name in ['S','N']): 
			h = self.mesh.y.cells[self.mesh.j(patch[0])] - self.mesh.y.cells[self.mesh.j(patch[1])]
		self.data = np.concatenate([self.data, np.ones(len(patch[0])), -np.ones(len(patch[1]))])
		self.B[patch[0]] += constant * h
	def matrix(self):
		matrix_size = len(self.mesh.x.cells) * len(self.mesh.y.cells)
		return csr_matrix((self.data, (self.row, self.col)), shape = (matrix_size, matrix_size))
		


if __name__ == '__main__':
	#x_faces = np.linspace(0,1,30)
	x_faces = np.linspace(0,1,5)
	xx = Axis(x_faces)
	# y_faces = np.concatenate([np.linspace(0, 0.8, 9), 1-np.logspace(np.log10(0.2), np.log10(1e-3), 21)[1:], [1]])
	#y_faces = np.concatenate([[0], np.logspace(np.log10(1e-3), np.log10(0.2), 11)[:-1], np.linspace(0.2, 1, 5)])
	y_faces = np.linspace(0,1,5)
	yy = Axis(y_faces)
	mesh = Structured2DMesh(xx, yy)
	'''
	import matplotlib.pyplot as plt
	plt.scatter(x_faces, np.zeros(len(x_faces)))
	plt.scatter(np.zeros(len(y_faces)), y_faces)
	plt.show()
	'''
	a = CellScalar(1, mesh=mesh)

	#def f(x,y): return y/np.sqrt(x)
	def f(x, y): return np.array([x+y,x-y])
	b = CellVector(f, mesh=mesh)

	#P = np.array([1,8,15,22,29,36])
	#Q = np.array([7,8,9,10,11,12,13])
	
	A = fvMatrix('lap', mesh=mesh).matrix()
	
	bc = fvBoundary(mesh=mesh)
	bc.add_constant_dirichlet('S', 0)
	bc.add_constant_dirichlet('N', 0)
	bc.add_constant_neumann('W', 1)
	bc.add_constant_neumann('E', 1)
	A = A + bc.matrix()
	T = linalg.spsolve(A, bc.B)
	print T.reshape(mesh.x.num+2, mesh.y.num+2)
	
