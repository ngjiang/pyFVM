import numpy as np


def in_ddt(dt):
	return np.array([
			np.ones ((n_nodes-2, n_nodes-2))/dt,
			np.zeros((n_nodes-2, n_nodes-2)),
			np.zeros((n_nodes-2, n_nodes-2)),
			np.zeros((n_nodes-2, n_nodes-2)),
			np.zeros((n_nodes-2, n_nodes-2)),
		])

def laplacian():
	return np.array([
			-(2./dx[1:-1,:-1]/dx[1:-1,1:] + 2./dy[:-1,1:-1]/dy[1:,1:-1]),
			2./dx[1:-1,:-1]/(dx[1:-1,:-1]+dx[1:-1,1:]),
			2./dx[1:-1,1:]/(dx[1:-1,:-1]+dx[1:-1,1:]),
			2./dy[:-1,1:-1]/(dy[:-1,1:-1]+dy[1:,1:-1]),
			2./dy[1:,1:-1]/(dy[:-1,1:-1]+dy[1:,1:-1]),
		])


