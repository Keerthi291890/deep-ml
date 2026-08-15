import numpy as np
from collections import Counter


def dummy_classifier(y_train, n_test, strategy, constant=None):
    """
    Produce baseline predictions of length n_test using the given strategy.
    Returns a Python list of predicted labels.
    """
    res = []

    if strategy == "most_frequent":
        most_freq = Counter(y_train).most_common(1)[0][0]

        for i in range(n_test):
            res.append(most_freq)

    elif strategy == "constant":
        for i in range(n_test):
            res.append(constant)

    elif strategy == "uniform":
        sorted_unique = np.unique(y_train)
        k = len(sorted_unique)

        for i in range(n_test):
            res.append(sorted_unique[i % k])

    elif strategy == "stratified":
        sorted_classes = np.unique(y_train)
        counts = Counter(y_train)

        # Ideal number of predictions for each class
        ideal = []

        for c in sorted_classes:
            freq = counts[c] / len(y_train)
            ideal.append(n_test * freq)

        # Floor allocation
        alloc = [int(x) for x in ideal]

        # Fractional parts
        frac = [ideal[i] - alloc[i] for i in range(len(ideal))]

        # Number of predictions still left
        left_pred = n_test - sum(alloc)

        # Distribute remaining predictions
        while left_pred > 0:
            max_index = 0

            for i in range(len(frac)):
                if frac[i] > frac[max_index]:
                    max_index = i

            alloc[max_index] += 1
            frac[max_index] = -1
            left_pred -= 1

        # Output grouped by sorted class
        for i in range(len(sorted_classes)):
            for _ in range(alloc[i]):
                res.append(sorted_classes[i])

    else:
        raise ValueError("Unknown strategy")

    return res