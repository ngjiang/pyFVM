# Square/rectangular cavity:
#      1 t
#      +---+
#    l | i | r
#      +---+ 1
#    O   b
#
# b = bottom, t = top, l = left, r = right, i = internal region
#
# Solve equations:
# (1): div(U) == 0
# (2): ddt(U) + div(phi, U) + grad(pd) == 1/Re div(tau) + T * e_y
# (3): ddt(T) + div(phi, T) == 1/(Re * Pr) laplacian(T)
# phi is the flux of U: phi = U 

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve, inv
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
matplotlib.rcParams.update({'font.size': 18})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Non-dimensionalized numbers, U0 = \sqrt{\Delta \rho / \rho_0 * g * L_0}
Ra = 1.e6                    # Rayleigh number
Pr = 0.7                     # Prandtl number 
Re = (Ra * Pr) ** 0.5        # Reynolds number

# Solution parameters
set_nCorr = 3
set_nNonOrtho = 0

# Control parameters
final_time = 10.0
time_step = 0.1

# Mesh parameter
n_nodes = 21

def export_mesh():
	print 'Exporting mesh into ./mesh.pdf...'
	plt.figure(figsize = (5,4))
	plt.scatter(x,y,c='b')
	plt.scatter(xs,ys,c='y')
	plt.plot([0,1,1,0,0],[0,0,1,1,0],c='k')
	plt.axis('equal')
	#plt.xlim(-0.1,1.1)
	plt.xlabel(r'$x$')
	plt.ylim(-0.05,1.05)
	plt.ylabel(r'$y$')
	#plt.savefig('mesh.pdf',bbox_inches='tight')
	#plt.show()
#export_mesh()

# Velocity and temperature are defined on x,y
# Pressure is defined on x,y, using Rhie-Chow interpolation

def piso():
	# Non-dimensionalized space
	x_axis = np.linspace(0., 1., num = n_nodes)
	y_axis = x_axis[:]
	
	x,y = np.meshgrid(x_axis, y_axis)      # f[i_y,j_x]
	
	import fvUtilities as fv
	fv.init_mesh((x,y))
	
	
	node_indices = np.arange(n_nodes*n_nodes).reshape((n_nodes, n_nodes))
	
	# Calculate dx, dy
	# Indices: i(w) = I(P)-1, i(e) = I(P)
	#          i(s) = I(W)-1, i(n) = I(N) 
	dx = x[:,1:] - x[:,:-1]
	dy = y[1:,:] - y[:-1,:]

	# Define the ICs
	T = np.zeros((n_nodes, n_nodes))                    # Reference temperature
	U = np.array([np.zeros((n_nodes, n_nodes))]*2)      # Zero velocity
	pd = np.zeros((n_nodes, n_nodes))               # Zero driving pressure
	# Initialize with the BC
	T[:, 0] +=  0.5
	T[:,-1] += -0.5
	# Dirichlet BC for U: no modification
	
	# Define the BCs
	print 'Defining boundary conditions...'
	# Boundary conditions
	# Nodes on boundary, no intersection occurs
	b_nodes = {
		# boundary nodes, for Dirichlet and Neumann BCs
		'bottom'  : node_indices[ 0,1:-1],
		'top'     : node_indices[-1,1:-1],
		'left'    : node_indices[:, 0],
		'right'   : node_indices[:,-1],
		# bottom and top
		'b,t'     : np.concatenate((node_indices[ 0,1:-1], node_indices[-1,1:-1])),
		# left and right
		'l,r'     : np.concatenate((node_indices[:, 0], node_indices[:, -1])),
		# neighbour nodes, for Neumann BCs
		'bottom_N': node_indices[ 1,1:-1],
		'top_N'   : node_indices[-2,1:-1],
		'left_N'  : node_indices[:, 1],
		'right_N' : node_indices[:,-2],
		'b,t_N'	  : np.concatenate((node_indices[ 1,1:-1], node_indices[-2,1:-1])),
		'l,r_N'   : np.concatenate((node_indices[:, 1], node_indices[:, -2])),
		}
	b_values = {
		'T_l,r'   : np.concatenate((np.ones(n_nodes)*0.5, np.ones(n_nodes)*-0.5)),
		'0_l,r'   : np.concatenate((np.zeros(n_nodes), np.zeros(n_nodes))),
		'0_b,t'   : np.concatenate((np.zeros(n_nodes-2), np.zeros(n_nodes-2))),
		'1_l,r'   : np.concatenate((np.ones(n_nodes), np.ones(n_nodes))),
		'1_b,t'   : np.concatenate((np.ones(n_nodes-2), np.ones(n_nodes-2))),
		}
	
	
	# Initialization of equation coefficients
	print 'Initializing equation coefficients...'
	# Internal region
	# Number of internal nodes
	n_in_nodes = (n_nodes-2)*(n_nodes-2)
	in_row, in_col = fv.internal_rows_columns()
	# coefficients in the internal region
	in_coef = {
		'ddt(.)*dt':
		fv.in_matrix_coef('ddt(.)*dt',()),
		'laplacian(.)':
		fv.in_matrix_coef('laplacian(.)',())
		}
	
	
	# Define mass matrix (dV)
	# Only in internal region!! (no boundary contribution)
	AdV = csr_matrix((fv.in_matrix_coef('dV',()).reshape(n_in_nodes*5), (in_row, in_col)), 
		shape=(n_nodes*n_nodes,n_nodes*n_nodes), 
		dtype = np.float64)

	print 'Initialization succeeded.'
	
	print 
	print 'Starting time loop...'
	
	# Time loop
	time = 0.	
	while time < final_time:
		print 'Time = ', time
		
		# Point old data to index n 
		Un = U
		pdn = pd
		Tn = T
		
		## Update 1.1: update phi with function fv_matrix('div(phi, .)', (x,y,U))
		in_coef['div(phi, .)'] = fv.in_matrix_coef('div(phi, .)', fv.flux(U))
	
		# Source terms in the internal region
		'''
		grad_pd = np.array([
				(pd[1:-1,2:] - pd[1:-1,:-2]) / (x[1:-1,2:] - x[1:-1,:-2]),
				(pd[2:,1:-1] - pd[:-2,1:-1]) / (y[2:,1:-1] - y[:-2,1:-1]),
			])
		'''
		grad_pd = fv.in_source('grad', pd)
	
		# (UEqn): ddt(U) + div(phi,U) - 1/Re * laplacian(U) == -grad(pd)
		# (UEqn): A U = H - grad(p)
		# phi prepared for upwind scheme
		# Indices: i(w) = I(P)-1, i(e) = I(P)
		#          i(s) = I(W)-1, i(n) = I(N) 
		#
		# Build the matrix AU for the first time
		# Coefficients for u and v are in common
		row  = np.concatenate((in_row, b_nodes ['b,t'], b_nodes ['l,r']))
		col  = np.concatenate((in_col, b_nodes ['b,t'], b_nodes ['l,r']))
		sum_in_coef = (\
			in_coef['ddt(.)*dt']/time_step\
			+ in_coef['div(phi, .)'] - in_coef['laplacian(.)']/Re
			).reshape(n_in_nodes*5)
		coef = np.concatenate((sum_in_coef, b_values['1_b,t'], b_values['1_l,r']))
		AU = csr_matrix((coef, (row, col)), 
			shape=(n_nodes*n_nodes,n_nodes*n_nodes), 
			dtype = np.float64)
		AP = AU.diagonal()      # Diagonal part of AU, coefficients for U_P
		#	
		# Build (UEqn).x
		# Remark: np.concatenate(()) takes at most 3 arguments
		# Initialize the source 
		Sx = np.zeros((n_nodes, n_nodes), dtype = np.float64)
		Sx[1:-1, 1:-1] += Un[0, 1:-1, 1:-1]/time_step + grad_pd[0]
		# Zeros on the whole boundary for no slip BC
		# Solve u
		ur = spsolve(AU, Sx.reshape(n_nodes*n_nodes))
		Hrx = ( \
			Un[0,:,:].reshape(n_nodes*n_nodes)/time_step \
			- AU * ur + AP * ur \
			).reshape(n_nodes, n_nodes)
		ur = ur.reshape((n_nodes,n_nodes))
		#
		# Build (UEqn).y
		# Initialize the source 
		Sy = np.zeros((n_nodes, n_nodes), dtype = np.float64)
		Sy[1:-1, 1:-1] += Un[1, 1:-1, 1:-1]/time_step + grad_pd[1] + T[1:-1,1:-1]
		# Zeros on the whole boundary for no slip BC
		# Solve v
		vr = spsolve(AU, Sy.reshape(n_nodes*n_nodes))
		# Hry Updateified from icoFoam, including T in internal region
		in_T = np.zeros((n_nodes, n_nodes))
		in_T[1:-1, 1:-1] += T[1:-1, 1:-1]
		Hry = (\
			Un[1,:,:].reshape(n_nodes*n_nodes)/time_step \
			- AU * vr + AP * vr \
			+ in_T.reshape(n_nodes*n_nodes) \
			).reshape(n_nodes, n_nodes)
		vr = vr.reshape((n_nodes,n_nodes))
		#
		# Construct vector Ur
		Ur = np.array([ur, vr])
		Hr = np.array([Hrx, Hry])
		# Copy Tn to Tr
		Tr = np.zeros((n_nodes, n_nodes)) + Tn
	
		# PISO correction
		nCorr = set_nCorr + 1
		while nCorr:
			rAP = (1./AP).reshape(n_nodes, n_nodes)
			HbyA = Hr * rAP
			nNonOrtho = set_nNonOrtho + 1
			while nNonOrtho:
				# Build (pEqn) and solve pdr
				# (pEqn): laplacian(rAP, pd) == div(HbyA)
				row  = np.concatenate((\
					in_row, \
					b_nodes ['b,t'], b_nodes ['b,t'],\
					b_nodes ['l,r'], b_nodes ['l,r'],\
					## Update 0.2: reference integral(pdr)
					[node_indices[n_nodes/2,n_nodes/2]]*(n_nodes*n_nodes)
					))
				col  = np.concatenate((\
					in_col, \
					b_nodes ['b,t'], b_nodes ['b,t_N'],\
					b_nodes ['l,r'], b_nodes ['l,r_N'],\
					## Update 0.2: reference integral(pdr)
					np.arange(n_nodes*n_nodes)
					))
				sum_in_coef = in_coef['laplacian(.)'] * rAP[1:-1,1:-1]
				## Update 0.1: reference pdr
				sum_in_coef[:,n_nodes/2-1,n_nodes/2-1] = [0.,0.,0.,0.,0.]
				sum_in_coef = sum_in_coef.reshape(n_in_nodes*5)
				
				coef = np.concatenate((\
					sum_in_coef, \
					b_values['1_b,t'], b_values['1_b,t']*-1, \
					b_values['1_l,r'], b_values['1_l,r']*-1, \
					## Update 0.2: reference integral(pdr)
					np.ones(n_nodes*n_nodes)
					))
					
				Apd = csr_matrix((coef, (row, col)), 
					shape=(n_nodes*n_nodes,n_nodes*n_nodes), 
					dtype = np.float64)
				Spd = np.zeros((n_nodes, n_nodes))
				Spd[1:-1, 1:-1] = fv.in_source('div', HbyA)
				# Compatitibility correction
				# integral(div(HbyA), dV) == integral(HbyA, dS)_B = 0 
				# Spd = Spd - int(Spd)
				Spd[1:-1, 1:-1] -= (AdV * Spd.reshape(n_nodes*n_nodes)).sum()
				## Update 0.1/0.2: reference pdr / reference intergral(pdr) 
				Spd[n_nodes/2,n_nodes/2] = 0.
				pdr = spsolve(Apd, Spd.reshape(n_nodes*n_nodes)).reshape(n_nodes, n_nodes)
				
				nNonOrtho -= 1
			# Correct Ur with pdr
			
			grad_pdr = fv.in_source('grad', pdr) 
			Ur = np.zeros((2,n_nodes,n_nodes))
			Ur[:,1:-1, 1:-1] = HbyA[:,1:-1, 1:-1] - rAP[1:-1, 1:-1] * grad_pdr
			
			# Rebuild phi from Ur
			in_coef['div(phi, .)'] = fv.in_matrix_coef('div(phi, .)', fv.flux(Ur))
			# Rebuild (UEqn), without solving it
			row  = np.concatenate((in_row, b_nodes ['b,t'], b_nodes ['l,r']))
			col  = np.concatenate((in_col, b_nodes ['b,t'], b_nodes ['l,r']))
			sum_in_coef = (\
				in_coef['ddt(.)*dt']/time_step\
				+ in_coef['div(phi, .)'] - in_coef['laplacian(.)']/Re
				).reshape(n_in_nodes*5)
			coef = np.concatenate((sum_in_coef, b_values['1_b,t'], b_values['1_l,r']))
			# Rebuild AU
			AU = csr_matrix((coef, (row, col)), 
				shape=(n_nodes*n_nodes,n_nodes*n_nodes), 
				dtype = np.float64)
			AP = AU.diagonal()      # Diagonal part of AU, coefficients for U_P
			# Rebuild Hr
			ur, vr = Ur.reshape(2,n_nodes*n_nodes)
			in_Tr = np.zeros((n_nodes, n_nodes))
			in_Tr[1:-1, 1:-1] += Tr[1:-1, 1:-1]
			Hr = np.array([
				( \
				Un[0,:,:].reshape(n_nodes*n_nodes)/time_step \
				- AU * ur + AP * ur \
				),
				(\
				Un[1,:,:].reshape(n_nodes*n_nodes)/time_step \
				- AU * vr + AP * vr \
				+ in_Tr.reshape(n_nodes*n_nodes) \
				),
				]).reshape(2,n_nodes, n_nodes)
			#
			# Build (TEqn) and solve
			row  = np.concatenate((in_row, b_nodes ['b,t'], b_nodes ['b,t'], b_nodes ['l,r']))
			col  = np.concatenate((in_col, b_nodes ['b,t'], b_nodes ['b,t_N'], b_nodes ['l,r']))
			sum_in_coef = (\
				in_coef['ddt(.)*dt']/time_step\
				+ in_coef['div(phi, .)'] - in_coef['laplacian(.)']/Re/Pr
				).reshape(n_in_nodes*5)
			coef = np.concatenate((sum_in_coef, b_values['1_b,t'], \
				b_values['1_b,t']*-1, b_values['1_l,r']))
			AT = csr_matrix((coef, (row, col)), 
				shape=(n_nodes*n_nodes,n_nodes*n_nodes), 
				dtype = np.float64)
			ST = np.zeros((n_nodes, n_nodes), dtype = np.float64)
			ST[:,0] = np.ones(n_nodes)*0.5
			ST[:,-1] = np.ones(n_nodes)*-0.5
			ST[1:-1, 1:-1] += Tn[1:-1, 1:-1]/time_step
			Tr = spsolve(AT, ST.reshape(n_nodes*n_nodes)).reshape(n_nodes, n_nodes)
			
			#plt.contourf(x,y,Tr,20)
			#plt.axis('equal')
			#plt.show()	
			
			nCorr -= 1
		# Update the final states
		U = Ur
		pd = pdr
		T = Tr
		
		time+=time_step
	
	plt.figure()
	ax = plt.gca()
	
	extent = np.min(x), np.max(x), np.min(y), np.max(y)
	ct = plt.contourf(x,y,T,20, 
		#cmap=plt.cm.viridis, 
		alpha=.9, interpolation='bilinear', extent=extent)
	#plt.quiver(x, y, U[0], U[1], units='width')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	plt.axis('equal')
	
	# create an axes on the right side of ax. The width of cax will be 5%
	# of ax and the padding between cax and ax will be fixed at 0.05 inch.
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(ct, cax=cax)
	plt.tight_layout()
	plt.show()
	#plt.savefig('./T.pdf',bbox_inches='tight')
	
if __name__ == '__main__':
	n_nodes = 65
	final_time = 50.
	piso()

