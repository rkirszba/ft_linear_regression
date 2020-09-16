import numpy as np
import matplotlib.pyplot as plt
from stat_utils import mean, standard_deviation


class FTLinearRegression():

    def __init__(self, alpha=0.001, max_iter=10000, add_intercept=True, convergence_threshold=0.0000001):
        self._alpha = alpha
        self._max_iter = max_iter
        self._intercept = add_intercept
        self._convergence_threshold = convergence_threshold
        self._mean = 0.
        self._sigma = 1.
        self._thetas = np.zeros((1, 2))
        self._costs = []

    @staticmethod    
    def _add_intercept(X):
        intercept = np.full((X.shape[0], 1), 1)
        return np.append(intercept, X, axis=1)
    
    def _normalize(self, X):
        return (X - self._mean) / self._sigma if self._sigma != 0 else X - self._mean

    def _compute_cost(self, X, y):
        return (1 / (2 * X.shape[0])) * np.sum(np.square(self._predict(X) - y))

    def _predict(self, X):
        return np.dot(X, self._thetas.T)
    
    def predict(self, X):
        X = self._normalize(X)
        if self._intercept:
            X = FTLinearRegression._add_intercept(X)
        return self._predict(X)

    def fit(self, X, y):
        self._mean = mean(X)
        self._sigma = standard_deviation(X)
        X = self._normalize(X)
        if self._intercept:
            X = FTLinearRegression._add_intercept(X)
        m = X.shape[0]
        for _ in range(self._max_iter):
            y_hat = self._predict(X)
            tmp_theta0 = (self._alpha / m) * np.sum(y_hat - y)
            tmp_theta1 = (self._alpha / m) * np.sum((y_hat - y) * X[:,1].reshape((-1, 1)))
            self._thetas[0][0] -= np.squeeze(tmp_theta0)
            self._thetas[0][1] -= np.squeeze(tmp_theta1)
            self._costs.append(np.squeeze(self._compute_cost(X, y)))
            if len(self._costs) > 1:
                diff_cost = self._costs[-1] - self._costs[-2]
                if diff_cost < 0 and abs(diff_cost / self._costs[-1]) < self._convergence_threshold:
                    break
        return self

    def plot_learning_curve(self):
        plt.plot(self._costs)
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.show()