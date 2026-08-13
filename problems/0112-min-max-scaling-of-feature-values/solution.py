import numpy as np
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    minimum = min(x)
    maximum = max(x)
    x_norm = []
    for i in x:
        temp = (i-minimum)/(maximum-minimum)
        x_norm.append(temp)
    return x_norm
    