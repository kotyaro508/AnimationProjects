import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, 2, 4])
b = np.array([15, 20, 25])


def func(t, x, alpha):
    f = np.zeros(3)
    f[0] = -(
        (a[2] / a[0]) ** 2
        * (alpha[2] * np.sin(x[2]) ** 2 - alpha[0] * np.sin(x[2]) * np.cos(x[2]))
        + (a[1] / a[0]) ** 2
        * (alpha[2] * np.cos(x[1]) ** 2 - alpha[1] * np.sin(x[1]) * np.cos(x[1]))
        + alpha[2] * (1 + b[2] / (a[0] ** 2))
    )
    f[1] = -(
        (a[0] / a[1]) ** 2
        * (alpha[0] * np.sin(x[0]) ** 2 - alpha[1] * np.sin(x[0]) * np.cos(x[0]))
        + (a[2] / a[1]) ** 2
        * (alpha[0] * np.cos(x[2]) ** 2 - alpha[2] * np.sin(x[2]) * np.cos(x[2]))
        + alpha[0] * (1 + b[0] / (a[1] ** 2))
    )
    f[2] = -(
        (a[1] / a[2]) ** 2
        * (alpha[1] * np.sin(x[1]) ** 2 - alpha[2] * np.sin(x[1]) * np.cos(x[1]))
        + (a[0] / a[2]) ** 2
        * (alpha[1] * np.cos(x[0]) ** 2 - alpha[0] * np.sin(x[0]) * np.cos(x[0]))
        + alpha[1] * (1 + b[1] / (a[2] ** 2))
    )
    return f


def rk4(phi_span, x0, options, alpha):
    npoints = options["npoints"]
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
    return T, X


phi_span = np.array([0.0, 2 * np.pi])
x0 = np.array([0.0, 0.0, 0.0])
options = {"npoints": 2000}
alpha = np.array([4.0, 2.0, 1.0])
alpha = alpha / np.linalg.norm(alpha)
T, X = rk4(phi_span, x0, options, alpha)
plt.figure(figsize=(15, 8))
plt.xlabel("phi", size=17)
plt.ylabel("- theta", size=17)
plt.grid(True, linestyle=":", alpha=1, color="black")
plt.plot(T, -X[0, :], lw=1.5, color="blue", label="theta_1")
plt.plot(T, -X[1, :], lw=1.5, color="orange", label="theta_2")
plt.plot(T, -X[2, :], lw=1.5, color="green", label="theta_3")
plt.legend(loc=2, prop={"size": 15})
