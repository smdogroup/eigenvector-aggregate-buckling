import numpy as np
from scipy.linalg import eigh, expm
import matplotlib.pylab as plt


def func_exact(rho, A, B, D):
    """
    Compute h directly
    """

    Binv = np.linalg.inv(B)
    exp = expm(-rho * np.dot(A, Binv))
    h = np.trace(np.dot(D, np.dot(Binv, exp))) / np.trace(exp)

    return h


def func(rho, A, B, D, N=None):
    """
    h = tr(D * B^{-1} * exp(- rho * A * B^{-1})/ tr(exp(- rho * A * B^{-1}))
    """
    if N is None:
        N = A.shape[0]

    # Compute the eigenvalues of the generalized eigen problem
    lam, Q = eigh(A, B)

    eta = np.exp(-rho * (lam - np.min(lam)))
    eta = eta / np.sum(eta)

    h = 0.0
    for i in range(N):
        h += eta[i] * np.dot(Q[:, i], np.dot(D, Q[:, i]))

    return h


rho = 10.0
M = 251
x = np.linspace(-1, 1, M)
y = np.linspace(-1, 1, M)
f1 = np.zeros((M, M))
f2 = np.zeros((M, M))
f3 = np.zeros((M, M))

D = np.array([[1, 0], [0, 0]])
B = np.eye(2)

for j in range(M):
    for i in range(M):
        A = np.array([[1.0 + x[i], -y[j]], [-y[j], 1.0 - x[i]]])

        f1[j, i] = func(rho, A, B, D)

        s = np.sqrt(x[i] ** 2 + y[j] ** 2)
        if s == 0.0:
            r = 1.0
        else:
            r = np.tanh(rho * s) / s

        f2[j, i] = 0.5 * (1.0 - x[i] * r)

        # Compute v1
        if y[j] == 0.0:
            if x[i] > 0.0:
                v1 = np.array(([0.0, 1.0]))
            else:
                v1 = np.array([1.0, 0.0])
        else:
            v1 = np.array(([-(x[i] - s) / y[j], 1.0]))
        v1 = v1 / np.sqrt(np.dot(v1, v1))

        f3[j, i] = v1[0] ** 2


plt.figure()
plt.contourf(x, y, f1)

plt.figure()
plt.contourf(x, y, f2)

plt.figure()
plt.contourf(x, y, f3)
plt.show()
