import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    inter = np.ones((X.shape[0],))
    X_train = np.column_stack((X,inter))
    weights = np.hstack((weights, bias))
    X_final = X_train[batch_indices]
    y_final = y[batch_indices]
    m = X_final.shape[0]
    y_hat = X_final @ weights
    error = y_hat - y_final
    grad = X_final.T @ error
    weights = weights - lr*2/m*grad
    
    return weights
