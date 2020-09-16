from math import sqrt
import numpy as np
from stat_utils import mean

def mean_squared_error(y_true, y_pred):
    return np.squeeze((1 / y_true.shape[0]) * np.sum(np.square(y_true - y_pred)))

def root_mean_squared_error(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.squeeze((1 / y_true.shape[0]) * np.sum(np.abs(y_true - y_pred)))

def r2_score(y_true, y_pred):
    num = np.sum(np.square(y_true - y_pred))
    den = np.sum(np.square(y_true - mean(y_true)))
    return 1 - np.squeeze(num / den)