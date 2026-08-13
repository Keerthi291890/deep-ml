import numpy as np

def to_categorical(x, n_col=None):
	n = np.unique(x)
	if n_col is None:
		n_col = n.shape[0]
	rows = x.shape[0]
	x_encoded = np.zeros((rows, n_col))
	for i in range(rows):
	   x_encoded[i][x[i]] = 1
	return x_encoded

	