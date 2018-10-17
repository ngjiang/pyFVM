import numpy as np
# Initialize mesh (x,y)
# Save x,y,
def init_mesh(data):
	global x,y,nx_nodes,ny_nodes,dx,dy,node_indices
	x,y = data
	ny_nodes, nx_nodes = np.shape(x)
	# Calculate dx, dy
	# Indices: i(w) = I(P)-1, i(e) = I(P)
	#          i(s) = I(W)-1, i(n) = I(N) 
	dx = x[:,1:] - x[:,:-1]
	dy = y[1:,:] - y[:-1,:]
	node_indices = np.arange(ny_nodes*nx_nodes).reshape((ny_nodes, nx_nodes))
	return

def internal_rows_columns():
	return [
		# row = node_indices[1:-1, 1:-1], [P]*5
		np.array([
			[node_indices[1:-1, 1:-1]]*5
			]).reshape((ny_nodes-2)*(nx_nodes-2)*5),
		# col = [P, W, E, S, N]
		np.array([
			node_indices[1:-1, 1:-1],
			node_indices[1:-1,  :-2],
			node_indices[1:-1, 2:  ],
			node_indices[ :-2, 1:-1],
			node_indices[2:  , 1:-1],
			]).reshape((ny_nodes-2)*(nx_nodes-2)*5)
		]

# Calculate internal matrix (no )
def in_matrix_coef(operator_name, data):
	# Euler scheme
	if operator_name in ['ddt(.)*dt','dt*ddt']:
		return np.array([
			np.ones ((ny_nodes-2, nx_nodes-2)),    # P
			np.zeros((ny_nodes-2, nx_nodes-2)),    # W
			np.zeros((ny_nodes-2, nx_nodes-2)),    # E
			np.zeros((ny_nodes-2, nx_nodes-2)),    # S
			np.zeros((ny_nodes-2, nx_nodes-2)),    # N
			])
	# Central scheme
	if operator_name in ['laplacian(.)', 'laplacian', 'lap']:
		return np.array([
			-(2./dx[1:-1,:-1]/dx[1:-1,1:] + 2./dy[:-1,1:-1]/dy[1:,1:-1]),    # P
			2./dx[1:-1,:-1]/(dx[1:-1,:-1]+dx[1:-1,1:]),                      # W
			2./dx[1:-1,1:]/(dx[1:-1,:-1]+dx[1:-1,1:]),                       # E
			2./dy[:-1,1:-1]/(dy[:-1,1:-1]+dy[1:,1:-1]),                      # S
			2./dy[1:,1:-1]/(dy[:-1,1:-1]+dy[1:,1:-1]),                       # N
			])
	# Upwind scheme
	if operator_name in ['div(phi, .)', 'div(phi,.)', 'u.grad']:
		phi_x, phi_y = data
		# phi_x defined on ((x[:,:-1]+x[:,1:])/2, y)
		# phi_y defined on (x, (y[:-1,:]+y[1:,:])/2)
		#
		# Upwind phi
		phi = {
			'x+':
			np.array([phi_x, np.zeros((ny_nodes, nx_nodes-1))]).max(0),
			'x-':
			np.array([phi_x, np.zeros((ny_nodes, nx_nodes-1))]).min(0),
			'y+':
			np.array([phi_y, np.zeros((ny_nodes-1, nx_nodes))]).max(0),
			'y-':
			np.array([phi_y, np.zeros((ny_nodes-1, nx_nodes))]).min(0),
			}
		return np.array([
			phi['x+'][1:-1,:-1]/dx[1:-1,:-1] - phi['x-'][1:-1,1:]/dx[1:-1,1:]
				+ phi['y+'][:-1,1:-1]/dy[:-1,1:-1] - phi['y-'][1:,1:-1]/dy[1:,1:-1],
			- phi['x+'][1:-1,:-1]/dx[1:-1,:-1],    # - phi_w/dx_w
			+ phi['x-'][1:-1,1:]/dx[1:-1,1:],      # + phi_e/dx_e
			- phi['y+'][:-1,1:-1]/dy[:-1,1:-1],    # - phi_s/dx_s
			+ phi['y-'][1:,1:-1]/dy[1:,1:-1],      # + phi_n/dx_n
			])
	# Mass matrix
	if operator_name in ['dV']:
		return np.array([
			(x[1:-1,2:]-x[1:-1,:-2])*(y[2:,1:-1]-y[:-2,1:-1])/4,    # P
			np.zeros((ny_nodes-2, nx_nodes-2)),                     # W
			np.zeros((ny_nodes-2, nx_nodes-2)),                     # E
			np.zeros((ny_nodes-2, nx_nodes-2)),                     # S
			np.zeros((ny_nodes-2, nx_nodes-2)),                     # N
			])
			
	return

def in_source(operator_name, data):
	# Gradient in source terms in internal region
	if operator_name in ['grad(.)','grad']:
		scalar = data
		return np.array([
			(scalar[1:-1,2:] - scalar[1:-1,:-2]) / (x[1:-1,2:] - x[1:-1,:-2]), 
			(scalar[2:,1:-1] - scalar[:-2,1:-1]) / (y[2:,1:-1] - y[:-2,1:-1]), 
			])
	# Divergence in source terms in internal region
	if operator_name in ['div', 'div(.)']:
		vector = data
		return \
			(vector[0,1:-1,2:] - vector[0,1:-1,:-2]) / (x[1:-1,2:] - x[1:-1,:-2]) \
			+(vector[1,2:,1:-1] - vector[1,:-2,1:-1]) / (y[2:,1:-1] - y[:-2,1:-1])

	return

# Define flux of a vector
def flux(data):
	ux, uy = data
	phi_x = (ux[:, :-1] + ux[:, 1:])/2        # defined on ((x[:,:-1]+x[:,1:])/2, y)
	phi_y = (uy[:-1, :] + uy[1:, :])/2        # defined on (x, (y[:-1,:]+y[1:,:])/2)
	return phi_x, phi_y

