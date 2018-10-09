# FVM in a 2D rectangle domain
# Upgraded from https://github.com/danieljfarrell/FVM

from __future__ import division
import collections
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse import dia_matrix
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
		self.cells = 0.5 * (self.faces[:-1] + self.faces[1:])
		self.num = len(self.cells)
		self.cell_widths = (self.faces[1:] - self.faces[:-1])

	''' Width of the cell at the specified index. '''
	def h(self, i):
		return self.cell_widths[i]

	''' Distance between centroids in the backward direction. '''
	def hm(self, i):
		if not check_index_within_bounds(i, 1, self.num-1):
			raise ValueError("hm index runs out of bounds")
		return (self.cells[i] - self.cells[i-1])

	''' Distance between centroids in the forward direction. '''
	def hp(self, i):
		if not check_index_within_bounds(i, 0, self.num-2):
			raise ValueError("hp index runs out of bounds")
		return (self.cells[i+1] - self.cells[i])

''' Class of 2D mesh cell-centered mesh defined by two axes x and y '''
class Structured2DMesh(object):
	def __init__(self, x, y):
		super(Structured2DMesh, self).__init__()
		self.x = x
		self.y = y
		# Construction of cells
		self.cells = np.arange(x.num * y.num).reshape(x.num, y.num)
		self.cell_j, self.cell_i = np.meshgrid(np.arange(y.num), np.arange(x.num))
		self.cell_i = self.cell_i.reshape(x.num * y.num)
		self.cell_j = self.cell_j.reshape(x.num * y.num)
		self.cell_x, self.cell_y = x.cells[self.cell_i], y.cells[self.cell_j]

		# Construction of faces
		self.faces = np.arange((x.num+1)*y.num + x.num*(y.num+1))
		## Faces to the direction of x, type of w and e
		self.faces_to_x = self.faces[:(x.num+1)*y.num].reshape(x.num+1, y.num)
		self.face_to_x_j, self.face_to_x_i = np.meshgrid(np.arange(y.num), np.arange(x.num+1))
		## Faces in direction of y, type of s and n
		self.faces_to_y = self.faces[(x.num+1)*y.num:].reshape(x.num, y.num+1)
		self.face_to_y_j, self.face_to_y_i = np.meshgrid(np.arange(y.num+1), np.arange(x.num))
		#
		self.face_x = np.concatenate([self.x.faces[self.face_to_x_i].reshape((x.num+1)*y.num), self.x.cells[self.face_to_y_i].reshape(x.num*(y.num+1))])
		self.face_y = np.concatenate([self.y.cells[self.face_to_x_j].reshape((x.num+1)*y.num), self.y.faces[self.face_to_y_j].reshape(x.num*(y.num+1))])
	# Define neighboring cells
	def W(self, P):
		return self.cells[self.cell_i[P]-1, self.cell_j[P]  ]
	def E(self, P):
		return self.cells[self.cell_i[P]-1, self.cell_j[P]  ]
	def S(self, P):
		return self.cells[self.cell_i[P],   self.cell_j[P]-1]
	def N(self, P):
		return self.cells[self.cell_i[P],   self.cell_j[P]+1]

	# Define neighboring faces
	def w(self, P):
		return self.faces_to_x[self.cell_i[P],   self.cell_j[P]  ]
	def e(self, P):
		return self.faces_to_x[self.cell_i[P]+1, self.cell_j[P]  ]
	def s(self, P):
		return self.faces_to_y[self.cell_i[P],   self.cell_j[P]  ]
	def n(self, P):
		return self.faces_to_y[self.cell_i[P],   self.cell_j[P]+1]

	# Define i and j
	def i(self, P):
		return self.cell_i[P]
	def j(self, P):
		return self.cell_j[P]



''' Representation of a variable defined at the cell centers. Provides interpolation functions to calculate the value at cell faces. '''
# http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
class CellVariable(np.ndarray):
	def __new__(cls, input_array, mesh = None):
		# Conversion of 'input_array' to array if it is actually a function of (x,y)
		if hasattr(input_array, '__call__'):
			input_array = input_array(mesh.cell_x, mesh.cell_y)
		# Conversion of 'input_array' to array if it is actually just a constant
		else :
			try:
				len(input_array)
			except:
				input_array = input_array * np.ones(mesh.x.num * mesh.y.num)

		obj = np.asarray(input_array).view(cls)
		obj.mesh = mesh
		return obj

	def __array_finalize__(self, obj):
		if obj is None: return
		self.mesh = getattr(obj, 'mesh', None)
		self.__get_items__ = getattr(obj, '__get_items__', None)

	''' Linear interpolation of the cell value at the west face '''
	def w(self, P):
		x, y = self.mesh.x, self.mesh.y
		iP = self.mesh.i(P); iW = iP - 1
		W = self.mesh.cells[iW, self.mesh.cell_j[P]]
		return (x.h(iP)*self[W] + x.h(iW)*self[P])/(2*x.hm(iP))
	''' Linear interpolation of the cell value at the east face '''
	def e(self, P):
		x, y = self.mesh.x, self.mesh.y
		iP = self.mesh.i(P); iE = iP + 1
		E = self.mesh.cells[iE, self.mesh.cell_j[P]]
		return (x.h(iP)*self[E] + x.h(iE)*self[P])/(2*x.hp(iP))
	''' Linear interpolation of the cell value at the south face '''
	def s(self, P):
		x, y = self.mesh.x, self.mesh.y
		jP = self.mesh.j(P); jS = jP - 1
		S = self.mesh.cells[self.mesh.cell_i[P], jS]
		return (y.h(jP)*self[S] + y.h(jS)*self[P])/(2*y.hm(jP))
	''' Linear interpolation of the cell value at the north face '''
	def n(self, P):
		x, y = self.mesh.x, self.mesh.y
		jP = self.mesh.j(P); jN = jP + 1
		N = self.mesh.cells[self.mesh.cell_i[P], jN]
		return (y.h(jP)*self[N] + y.h(jN)*self[P])/(2*y.hp(jP))

	def on_faces(self):
		return FaceVariable(self)

''' Representation of a variable defined at the face centers. '''
class FaceVariable(np.ndarray):
	def __new__(cls, input_array, mesh = None):
		# Conversion of 'input_array' to array if it is actually a function of (x,y)
		if hasattr(input_array, '__call__'):
			input_array = input_array(mesh.face_x, mesh.face_y)
		# Conversion of 'input_array' to array if it is actually just a constant
		else :
			try:
				len(input_array)
			except:
				input_array = input_array * np.ones(mesh.x.num * mesh.y.num)

		obj = np.asarray(input_array).view(cls)
		obj.mesh = mesh
		return obj

	def __array_finalize__(self, obj):
		if obj is None: return
		self.mesh = getattr(obj, 'mesh', None)
		self.__get_items__ = getattr(obj, '__get_items__', None)

if __name__ == '__main__':
	#x_faces = np.linspace(0,1,30)
	x_faces = np.linspace(0,1,5)
	xx = Axis(x_faces)
	# y_faces = np.concatenate([np.linspace(0, 0.8, 9), 1-np.logspace(np.log10(0.2), np.log10(1e-3), 21)[1:], [1]])
	#y_faces = np.concatenate([[0], np.logspace(np.log10(1e-3), np.log10(0.2), 11)[:-1], np.linspace(0.2, 1, 5)])
	y_faces = np.linspace(0,1,6)
	yy = Axis(y_faces)
	mesh = Structured2DMesh(xx, yy)
	'''
	import matplotlib.pyplot as plt
	plt.scatter(x_faces, np.zeros(len(x_faces)))
	plt.scatter(np.zeros(len(y_faces)), y_faces)
	plt.show()
	'''
	a = CellVariable(1, mesh=mesh)

	#def f(x,y): return y/np.sqrt(x)
	def f(x, y): return x+y
	b = CellVariable(f, mesh=mesh)
