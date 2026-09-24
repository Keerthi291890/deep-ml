import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	hypo = X@ weights+bias
	y_hat = 1/(1+np.exp(-hypo))
	res = np.zeros(y_hat.size)
	for i in range(len(y_hat)):
		if y_hat[i]>=0.5:
			res[i] = 1
		else:
			res[i] = 0
	return res.astype(int)
	