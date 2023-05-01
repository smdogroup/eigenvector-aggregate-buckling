import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import mpmath as mp
import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import minimize


def rand_symm_mat(n=10, eig_low=0.1, eig_high=100.0, nrepeat=1):
    # Randomly generated matrix that will be used to generate the eigenvectors
    QRmat = -1.0 + 2 * np.random.uniform(size=(n, n))

    Q, _ = np.linalg.qr(QRmat, mode="complete")  # Construct Q via a Q-R decomposition

    if nrepeat == 1:
        lam = np.random.uniform(low=eig_low, high=eig_high, size=n)
    else:
        lam = np.hstack(
            (
                eig_low * np.ones(nrepeat),
                np.random.uniform(low=eig_low, high=eig_high, size=n - nrepeat),
            )
        )

    return np.dot(Q, np.dot(np.diag(lam), Q.T))  # Compute A = Q*Lambda*Q^{T}


def deriv(rho, A, B, D, Adot, Bdot, Ddot=None, ndvs=1):
    """
    Compute the forward mode derivative
    """

    # Compute the eigenvalues of the generalized eigen problem
    lam, Q = eigh(A, B)

    lam_min = np.min(lam)
    eta = np.exp(-rho * (lam - lam_min))
    trace = np.sum(eta)
    eta = eta / trace

    # compute the h value
    h = func(eta, Q, D, np.shape(A)[0])
    hdot = np.zeros(ndvs)

    for j in range(np.shape(A)[0]):
        for i in range(np.shape(A)[0]):
            Adot_q = Q[:, i].T @ Adot @ Q[:, j]
            Bdot_q = Q[:, i].T @ Bdot @ Q[:, j]
            Dq = Q[:, i].T @ D @ Q[:, j]
            Eij = precise(rho, trace, np.min(lam), lam[i], lam[j])

            if i == j:
                hdot += Eij * (Dq - h) * (Adot_q - lam[j] * Bdot_q)
            else:
                hdot += Eij * Dq * (Adot_q - lam[j] * Bdot_q)

            hdot -= eta[i] * Dq * Bdot_q

    for j in range(np.shape(A)[0]):
        if Ddot is not None:
            hdot += eta[j] * Q[:, j].T @ Ddot @ Q[:, j]

    return hdot


def precise(rho, trace, lam_min, lam1, lam2):
    """
    Compute the precise value of the E_{ij} term

        E_{ij} = exp(-rho * (lam1 - lam_min)) / trace

        if lam1 == lam2:
            E_{ij} = exp(-rho * (lam1 - lam_min)) / trace
        else:
            E_{ij} = (exp(-rho * (lam1 - lam_min)) - exp(-rho * (lam2 - lam_min))) / (lam1 - lam2) / trace
    """

    with mp.workdps(80):
        if lam1 == lam2:
            val = -rho * mp.exp(-rho * (lam1 - lam_min)) / trace
        else:
            val = (
                (mp.exp(-rho * (lam1 - lam_min)) - mp.exp(-rho * (lam2 - lam_min)))
                / (mp.mpf(lam1) - mp.mpf(lam2))
                / mp.mpf(trace)
            )
    return np.float64(val)


def func(eta, Q, D, N=None):
    """
    h = tr(eta * Q^T * D * Q)
    """
    if N is None:
        N = Q.shape[0]

    h = 0.0
    for i in range(N):
        h += eta[i] * Q[:, i].T @ D @ Q[:, i]

    return h


def deriv_approx(A, B, D, Adot, Bdot, Ddot=None, ndvs=1, rho=1.0, N=5):
    """
    Approximately compute the forward derivative

        first term:
            sum^N E_{ij} (Dq_{ij} + h delta_{ij})*(Aq_dot_{ij} - lam_{i} Bq_dot_{ij})

        second term:
            sum^N 2 * q_{i}^T * Adot * v_{j}
            sum^N q_{i}^T * Bdot * (u_{j} - lam_{j} * v_{j} - w_{j})

    """

    # solve the eigenvalue problem
    lam, Q = eigh(A, B)

    # compute eta
    eta = np.exp(-rho * (lam - np.min(lam)))
    trace = np.sum(eta)
    eta = eta / trace

    # count nonzero entries in the eta vector as self.Np
    N = np.count_nonzero(eta)

    C = B @ Q[:, :N]
    U, _ = np.linalg.qr(C)
    Z = np.eye(np.shape(A)[0]) - U @ U.T

    # compute eta
    eta = np.exp(-rho * (lam - np.min(lam)))
    trace = np.sum(eta)
    eta = eta / trace

    # compute the h value
    h = func(eta, Q, D, N)
    hdot = np.zeros(ndvs)

    # only compute the lower triangle of the matrix since it is symmetric
    for j in range(N):
        for i in range(j + 1):
            Adot_q = Q[:, i].T @ Adot @ Q[:, j]
            Bdot_q = Q[:, i].T @ Bdot @ Q[:, j]
            Dq = Q[:, i].T @ D @ Q[:, j]
            Eij = precise(rho, trace, np.min(lam), lam[i], lam[j])

            if i == j:
                hdot += Eij * (Dq - h) * (Adot_q - lam[j] * Bdot_q)
            else:
                hdot += Eij * Dq * (2 * Adot_q - (lam[i] + lam[j]) * Bdot_q)

    # compute second term in the derivative approximation
    for j in range(N):
        # solve the first linear system
        rhs1 = -eta[j] * D @ Q[:, j]
        uj = np.linalg.solve(B, rhs1)

        # solve the second linear system
        Ak = A - lam[j] * B
        rhs2 = rhs1
        Abar = Z.T @ Ak @ Z
        bbar = Z.T @ rhs2
        vj = Z @ np.linalg.solve(Abar, bbar)

        # solve the third linear system
        rhs3 = A @ uj
        bbar = Z.T @ rhs3
        wj = Z @ np.linalg.solve(Abar, bbar)

        # compute the contributions to the derivative
        hdot += 2 * Q[:, j].T @ Adot @ vj
        hdot += Q[:, j].T @ Bdot @ (uj - lam[j] * vj - wj)
        if Ddot is not None:
            hdot += eta[j] * Q[:, j].T @ Ddot @ Q[:, j]

    return hdot


# Set parameters
rho = 1000.0
N = 10
n = 100
dh = 1e-30
ndvs = 5

np.random.seed(12345)

x = 0.1 * np.ones(ndvs)
p = np.random.uniform(size=ndvs)

A = rand_symm_mat(n)
B = rand_symm_mat(n)
Adot = rand_symm_mat(n)
Bdot = rand_symm_mat(n)
Ddot = rand_symm_mat(n)
D = rand_symm_mat(n)

lam, Q = eigh(A, B)

eta = np.exp(-rho * (lam - np.min(lam)))
eta = eta / np.sum(eta)

times = []

ans = np.dot(deriv(rho, A, B, D, Adot, Bdot, Ddot,ndvs=ndvs), p)
ans_approx = np.dot(deriv_approx(A, B, D, Adot, Bdot,Ddot, ndvs=ndvs, rho=rho, N=N), p)

print("ans = ", ans)
print("ans_approx = ", ans_approx)
print("error = ", np.abs(ans - ans_approx) / np.abs(ans))
