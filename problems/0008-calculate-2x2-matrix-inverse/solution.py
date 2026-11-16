import numpy as np
import math
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    A = np.asarray(matrix)
    det = A[0][0]*A[1][1]-A[0][1]*A[1][0]
    if det==0: return None
    A_temp = np.array([[A[1][1],-A[0][1]],[-A[1][0],A[0][0]]])
    A_inv = np.round(1/det*(A_temp),1)
    return A_inv.tolist()
    pass