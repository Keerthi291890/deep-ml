import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Return a list where each element is the dot product of a row of 'a' with 'b'.
    # If the number of columns in 'a' does not match the length of 'b', return -1.
    a = np.asarray(a)
    b = np.asarray(b)
    (arows, acols) = a.shape
    (bcols, ) = b.shape
    if acols!=bcols: return -1
    res = np.zeros(arows)
    for i in range(arows):
        for j in range(bcols):
            res[i] += a[i][j]*b[j]
    return res.tolist()
    pass