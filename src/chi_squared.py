import numpy as np

def chi_squared(y_true, y_pred, y_error):
    diff = (y_true - y_pred)
    diffsq = diff * diff
    errorsq = y_error*y_error
    chi_square_n = diffsq/errorsq
    return np.average(chi_square_n)