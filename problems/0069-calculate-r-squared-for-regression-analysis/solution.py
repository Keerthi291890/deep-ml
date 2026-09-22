
import numpy as np

def r_squared(y_true, y_pred):
	y_mean = np.mean(y_true)
	sst = np.sum((y_true-y_mean)**2)
	sse = np.sum((y_true-y_pred)**2)
	r_square = 1-(sse/sst)
	return r_square
	pass
