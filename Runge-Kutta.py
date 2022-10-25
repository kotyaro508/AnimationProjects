import numpy as np


def rk4(phi_span, x0, npoints, alpha):

    T = np.linspace(phi_span[0], phi_span[1], npoints + 1)
    h = T[1] - T[0]

    X = np.empty((len(x0), npoints + 1))
    X[:, 0] = x0.copy()

    for k in range(npoints):
        k1 = func(T[k], X[:, k], alpha)
        k2 = func(T[k] + h / 2, X[:, k] + h * k1 / 2, alpha)
        k3 = func(T[k] + h / 2, X[:, k] + h * k2 / 2, alpha)
        k4 = func(T[k] + h, X[:, k] + h * k3, alpha)
        X[:, k + 1] = X[:, k] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return X
