import math
import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Standardization
    mean = []
    std = []

    rows = data.shape[0]
    columns = data.shape[1]

    for i in range(columns):
        total = 0

        for j in range(rows):
            total += data[j][i]

        mean.append(total / rows)

    # Calculate standard deviation
    for i in range(columns):
        var = 0

        for j in range(rows):
            var += (data[j][i] - mean[i]) ** 2

        std.append(math.sqrt(var / rows))

    standardized_data = np.zeros((rows, columns))

    # Apply standardization
    for i in range(rows):
        for j in range(columns):
            standardized_data[i][j] = (data[i][j] - mean[j]) / std[j]

    # Normalization
    min_values = []
    max_values = []

    for i in range(columns):
        min_values.append(min(data[:, i]))
        max_values.append(max(data[:, i]))

    normalized_data = np.zeros((rows, columns))

    for i in range(rows):
        for j in range(columns):
            normalized_data[i][j] = (
                (data[i][j] - min_values[j])
                / (max_values[j] - min_values[j])
            )

    return standardized_data, normalized_data