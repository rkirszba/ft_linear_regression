import numpy as np

def add_intercept(X):
    intercept = np.full((X.shape[0], 1), 1)
    return np.append(intercept, X, axis=1)

def estimate_price(X, thetas):
    return np.dot(X, thetas.T)