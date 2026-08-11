import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    We shuffle the dataset before splitting because the ordering of the dataset doesn't determine which examples to go into train, test, and validation.
    But we shouldn't shuffle the data in-place(not the original data). NumPy shuffles data in-place. So we've to create a separate suffled copy.
    rng = np.random.default_rng(seed) - to create random number
    data = rng.permutation(data) - returns a shuffled copy
    """
    rng = np.random.default_rng(seed)
    data = rng.permutation(data)
    n = data.shape[0] #No.of rows
    train_end = int(n*train_frac)
    val_end = train_end + int(n*validation_frac)
    train = np.array(data[:train_end])
    val = np.array(data[train_end:val_end])
    test = np.array(data[val_end:])
    return [train, val, test]
    pass