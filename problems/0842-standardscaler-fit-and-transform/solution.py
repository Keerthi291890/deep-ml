import numpy as np
import math
def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    #mean caluculation
    rows = X_train.shape[0]
    columns = X_train.shape[1]
    mean = []
    for i in range(columns):
        total = 0
        for j in range(rows):
           total+=X_train[j][i]
        mean.append(total/rows)
    #standard deviation calculation
    std = []
    for i in range(columns):
       total = 0
       for j in range(rows):
          total += (X_train[j][i]-mean[i])**2
       std.append(math.sqrt(total/rows))
    #transformation
    test_rows = X_test.shape[0]
    test_columns = X_test.shape[1]
    test_transform = np.zeros((test_rows, test_columns))
    for j in range(test_columns):
        for i in range(test_rows):
            if std[j]==0:
                test_transform[i][j] = (X_test[i][j]-mean[j])/1.0
            else:
                 test_transform[i][j] = (X_test[i][j] - mean[j])/std[j]
    return test_transform
    pass
