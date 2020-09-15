import numpy as np

def mean(X):
    return np.sum(X) / X.shape[0]

def variance(X):
    return np.sqrt((1 / X.shape[0]) * np.sum(np.square(X - mean(X))))

def normalize(X, mean, variance):
    return (X - mean) / variance if variance != 0 else X - mean

def add_intercept(X):
    intercept = np.full((X.shape[0], 1), 1)
    return np.append(intercept, X, axis=1)

def predict(X, thetas):
    return np.dot(X, thetas.T)

def compute_cost(X, y, thetas):
    return (1 / (2 * X.shape[0])) * np.sum(np.square(predict(X, thetas) - y))

def fit(X, y, thetas, alpha=0.001, max_iter=10000):
    costs = []
    m = X.shape[0]
    for _ in range(max_iter):
        y_hat = predict(X, thetas)
        tmp_theta0 = thetas[0][0] - (alpha / m) * np.sum(y_hat - y)
        tmp_theta1 = thetas[0][1] - (alpha / m) * np.sum((y_hat - y) * X[:,1].reshape((X.shape[0], 1)))
        thetas[0][0] = np.squeeze(tmp_theta0)
        thetas[0][1] = np.squeeze(tmp_theta1)
        costs.append(np.squeeze(compute_cost(X, y, thetas)))
        if len(costs) > 1:
            diff_cost = costs[-1] - costs[-2]
            if diff_cost < 0 and abs(diff_cost / costs[-1]) < 0.000001:
                break
    return thetas, costs