import matplotlib.pylab as plt
import numpy as np
import scienceplots
from scipy.linalg import eigh, expm
from icecream import ic

# Set parameters
rho = 1.0
n = 10
dh = 1e-30


def rand_symm_mat(n=10, eig_low=0.1, eig_high=100.0):
    # Randomly generated matrix that will be used to generate the eigenvectors
    QRmat = -1.0 + 2 * np.random.uniform(size=(n, n))

    Q, r = np.linalg.qr(QRmat, mode="complete")  # Construct Q via a Q-R decomposition
    lam = np.random.uniform(low=eig_low, high=eig_high, size=n)

    return np.dot(Q, np.dot(np.diag(lam), Q.T))  # Compute A = Q*Lambda*Q^{T}


np.random.seed(12345)

A = rand_symm_mat(n)
B = rand_symm_mat(n)
Adot = rand_symm_mat(n)
Bdot = rand_symm_mat(n)
D = rand_symm_mat(n)


def func2(rho, A, B, D):
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


def func2_1(rho, x, y):
    s = np.sqrt(x**2 + y**2)
    if s == 0.0:
        r = 1.0
    else:
        r = np.tanh(rho * s) / s

    f = 0.5 * (1.0 - x * r)
    return f


def deriv(rho, A, B, D, Adot, Bdot, N=None):
    """
    Compute the forward mode derivative
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

    E = np.zeros(A.shape)
    for j in range(A.shape[1]):
        for i in range(A.shape[0]):
            qDq = np.dot(Q[:, i], np.dot(D, Q[:, j]))
            qBq = np.dot(Q[:, i], np.dot(B, Q[:, j]))
            scalar = qDq - h * qBq

            if i == j or lam[i] == lam[j]:
                E[i, j] = -rho * eta[i] * scalar
            else:
                E[i, j] = ((eta[j] - eta[i]) / (lam[j] - lam[i])) * scalar

    mat = np.dot(Q.T, np.dot(Adot, Q)) - np.dot(
        np.diag(lam), np.dot(Q.T, np.dot(Bdot, Q))
    )

    G = np.dot(np.dot(Q.T, np.dot(D, Q)), np.dot(Q.T, np.dot(Bdot, Q)))

    hdot = np.trace(np.dot(E, mat))
    hdot -= np.trace(np.dot(G, np.diag(eta)))

    return hdot


f0 = func(rho, A, B, D)
f1 = func2(rho, A + 1j * dh * Adot, B + 1j * dh * Bdot, D)
fd = f1.imag / dh  # (f1 - f0) / dh

ans = deriv(rho, A, B, D, Adot, Bdot)

print("fd =  ", fd)
print("ans = ", ans)


rho = [10, 20]

M = 251
x = np.linspace(-1, 1, M)
y = np.linspace(-1, 1, M)
X, Y = np.meshgrid(x, y)
f1 = np.zeros((M, M, len(rho)))
f2 = np.zeros((M, M, len(rho)))
f3 = np.zeros((M, M))
lam1 = np.zeros((M, M))
lam2 = np.zeros((M, M))

D = np.array([[1, 0], [0, 0]])
B = np.eye(2)


for j in range(M):
    for i in range(M):
        for k in range(len(rho)):
            A = np.array([[1.0 + x[i], -y[j]], [-y[j], 1.0 - x[i]]])

            f1[j, i, k] = func2(rho[k], A, B, D)

            f2[j, i, k] = func2_1(rho[k], x[i], y[j])

            s = np.sqrt(x[i] ** 2 + y[j] ** 2)

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

            lam1[j, i] = 1 - s
            lam2[j, i] = 1 + s

with plt.style.context(["nature"]):
    text = ["(a)", "(b)", "(c)"
    ]
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.2), constrained_layout=True)
    ax[0].contourf(X, Y, f3[:, :], cmap="coolwarm", levels=20)
    ax[1].contourf(X, Y, f2[:, :, 0], cmap="coolwarm", levels=20)
    ax[2].contourf(X, Y, f2[:, :, 1], cmap="coolwarm", levels=20)
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[2].set_axis_off()
    for i in range(3):
      ax[i].text(
              0.5,
              -0.025,
              text[i],
              transform=ax[i].transAxes,
              va="top",
              ha="center",
              weight='bold',
            #   fontsize=9,
          )


plt.savefig("output/simple_example/seyranian_plot.pdf", bbox_inches="tight", pad_inches=0.0)
plt.show()
