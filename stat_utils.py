import numpy as np

def mean(X):
    return np.squeeze(np.sum(X) / X.shape[0])

def standard_deviation(X):
   return np.squeeze(np.sqrt((1 / X.shape[0]) * np.sum(np.square(X - mean(X)))))