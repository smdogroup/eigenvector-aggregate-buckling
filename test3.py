from icecream import ic
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import mpmath as mp
import numpy as np
from scipy.linalg import eig, eigh, expm
from scipy.optimize import minimize
import scipy.sparse


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


def softmax_ab(fun, rho, lam, lam_a, lam_b):
    """
    Compute the eta values
    """
    ic(rho, lam_a, lam_b)
    eta = np.zeros(len(lam), dtype=lam.dtype)
    for i in range(len(lam)):
        a = fun(rho * (lam[i] - lam_a))
        b = fun(rho * (lam[i] - lam_b))
        eta[i] = a - b
    ic(lam[:10])
    ic(eta[:10])
    return eta


# compute lam_a and lam_b based on the index
def compute_lam_index(lam, N_a, N_b):
    lam_a = lam[N_a]
    lam_b = lam[N_b]
    N_b += 1
    return lam_a, lam_b, N_a, N_b


def fun_h(eta, Q, D, N=None):
    """
    h = sum^N eta_i * q_i^T * D * q_i
    """

    eta = eta / np.sum(eta)

    if N is None:
        N = Q.shape[1]

    h = 0.0
    for i in range(N):
        h += eta[i] * Q[:, i].T @ D @ Q[:, i]

    return h


def Eij_a(fun, rho, trace, lam_min, lam1, lam2):
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
            val = -rho * fun(-rho * (lam1 - lam_min)) / trace
        else:
            val = (
                (fun(-rho * (lam1 - lam_min)) - fun(-rho * (lam2 - lam_min)))
                / (mp.mpf(lam1) - mp.mpf(lam2))
                / mp.mpf(trace)
            )
    return np.float64(val)


def Gij_a(fun, rho, trace, lam_min, lam1, lam2):
    with mp.workdps(80):
        if lam1 == lam2:
            val = -rho * lam1 * fun(-rho * (lam1 - lam_min)) / trace
        else:
            val = (
                lam1 * fun(-rho * (lam1 - lam_min))
                - lam2 * fun(-rho * (lam2 - lam_min))
            ) / ((mp.mpf(lam1) - mp.mpf(lam2)) * mp.mpf(trace))
    return np.float64(val)


def Eij_ab(fun, rho, trace, lam1, lam2, lam_a, lam_b):
    with mp.workdps(80):
        a1 = fun(rho * (mp.mpf(lam1) - mp.mpf(lam_a)))
        b1 = fun(rho * (mp.mpf(lam1) - mp.mpf(lam_b)))
        a2 = fun(rho * (mp.mpf(lam2) - mp.mpf(lam_a)))
        b2 = fun(rho * (mp.mpf(lam2) - mp.mpf(lam_b)))

        eta1 = a1 - b1
        eta2 = a2 - b2

        if lam1 == lam2:
            val = -rho * eta1 * (a1 + b1) / mp.mpf(trace)
            # val = 0.0
        else:
            val = (eta1 - eta2) / (mp.mpf(lam1) - mp.mpf(lam2)) / mp.mpf(trace)
    return np.float64(val)


def Gij_ab(fun, rho, trace, lam1, lam2, lam_a, lam_b):
    with mp.workdps(80):
        a1 = fun(rho * (mp.mpf(lam1) - mp.mpf(lam_a)))
        b1 = fun(rho * (mp.mpf(lam1) - mp.mpf(lam_b)))
        a2 = fun(rho * (mp.mpf(lam2) - mp.mpf(lam_a)))
        b2 = fun(rho * (mp.mpf(lam2) - mp.mpf(lam_b)))

        eta1 = a1 - b1
        eta2 = a2 - b2

        if lam1 == lam2:
            val = -rho * lam1 * eta1 * (a1 + b1) / mp.mpf(trace)
            # val = 0.0
        else:
            val = (
                (lam1 * eta1 - lam2 * eta2)
                / (mp.mpf(lam1) - mp.mpf(lam2))
                / mp.mpf(trace)
            )
    return np.float64(val)


def deriv_exict(
    rho,
    A,
    B,
    D,
    Adot,
    Bdot,
    lam,
    Q,
    eta,
    fun,
    Eij_fun,
    Gij_fun,
    lam_a=None,
    lam_b=None,
    Ddot=None,
):
    """
    Compute the forward mode derivative
    """

    trace = np.sum(eta)
    eta = eta / trace

    # compute the h value
    h = fun_h(eta, Q, D)
    hdot = 0.0

    for j in range(np.shape(A)[0]):
        for i in range(np.shape(A)[0]):
            Adot_q = Q[:, i].T @ Adot @ Q[:, j]
            Bdot_q = Q[:, i].T @ Bdot @ Q[:, j]
            qDq = Q[:, i].T @ D @ Q[:, j]

            if Eij_fun == Eij_a:
                Eij = Eij_fun(fun, rho, trace, np.min(lam), lam[i], lam[j])
                Gij = Gij_fun(fun, rho, trace, np.min(lam), lam[i], lam[j])
            else:
                Eij = Eij_fun(fun, rho, trace, lam[i], lam[j], lam_a, lam_b)
                Gij = Gij_fun(fun, rho, trace, lam[i], lam[j], lam_a, lam_b)

            if i == j:
                scale = qDq - h
            else:
                scale = qDq

            hdot += scale * (Eij * Adot_q - Gij * Bdot_q)

    for j in range(np.shape(A)[0]):
        hdot -= eta[j] * (Q[:, j].T @ D @ Q[:, j]) * (Q[:, j].T @ Bdot @ Q[:, j])

        if Ddot is not None:
            hdot += eta[j] * Q[:, j].T @ Ddot @ Q[:, j]

    return hdot


def deriv_approx(
    A,
    B,
    D,
    Adot,
    Bdot,
    lam,
    Q,
    eta,
    fun,
    Eij_fun,
    Gij_fun,
    N_a=0,
    N_b=5,
    lam_a=None,
    lam_b=None,
    Ddot=None,
    rho=1.0,
):
    # normalize the eta vector
    trace = np.sum(eta)
    eta = eta / trace

    # compute the h value
    h = fun_h(eta, Q, D)
    hdot = 0.0

    for i in range(N_b):
        for j in range(i + 1):
            qDq = Q[:, i].T @ D @ Q[:, j]
            qAdotq = Q[:, i].T @ Adot @ Q[:, j]
            qBdotq = Q[:, i].T @ Bdot @ Q[:, j]

            if Eij_fun == Eij_a:
                Eij = Eij_fun(fun, rho, trace, np.min(lam), lam[i], lam[j])
                Gij = Gij_fun(fun, rho, trace, np.min(lam), lam[i], lam[j])
            else:
                Eij = Eij_fun(fun, rho, trace, lam[i], lam[j], lam_a, lam_b)
                Gij = Gij_fun(fun, rho, trace, lam[i], lam[j], lam_a, lam_b)

            if i == j:
                scale = qDq - h
            else:
                scale = 2 * qDq

            hdot += scale * (Eij * qAdotq - Gij * qBdotq)

    # compute the orthogonal projector
    C = B @ Q[:, :N_b]
    U, _ = np.linalg.qr(C)

    factor = 0.99
    P = A - factor * lam[0] * B
    Pfactor = scipy.sparse.linalg.factorized(P)

    def preconditioner(x):
        y = Pfactor(x)
        t = np.dot(U.T, y)
        y = y - np.dot(U, t)
        return y

    preop = scipy.sparse.linalg.LinearOperator((n, n), matvec=preconditioner)

    # Z = np.eye(np.shape(A)[0]) - U @ U.T

    for j in range(N_b):
        Dq = D @ Q[:, j]
        bkr = -2.0 * eta[j] * Dq

        def matrix(x):
            y = A.dot(x) - lam[j] * B.dot(x)
            t = np.dot(U.T, y)
            y = y - np.dot(U, t)
            return y

        matop = scipy.sparse.linalg.LinearOperator((n, n), matvec=matrix)

        t = np.dot(U.T, bkr)
        bkr = bkr - np.dot(U, t)
        phi, _ = scipy.sparse.linalg.gmres(matop, bkr, tol=1e-10, atol=1e-15, M=preop)

        # Ak = A - lam[j] * B
        # Abar = Z.T @ Ak @ Z
        # bbar = Z.T @ (-2.0 * eta[j] * Dq)
        # phi = Z @ np.linalg.solve(Abar, bbar)

        hdot += Q[:, j].T @ (Adot - lam[j] * Bdot) @ phi

        hdot -= eta[j] * (Q[:, j].T @ D @ Q[:, j]) * (Q[:, j].T @ Bdot @ Q[:, j])

        if Ddot is not None:
            hdot += eta[j] * Q[:, j].T @ Ddot @ Q[:, j]

    return hdot


if __name__ == "__main__":
    rho = 5000.0
    m = 10
    dh = 1e-6

    np.random.seed(0)

    n = 100
    A = np.random.randn(n, n)
    A = A + A.T
    B = rand_symm_mat(n)
    D = rand_symm_mat(n)

    Adot = rand_symm_mat(n)
    Bdot = rand_symm_mat(n)

    A1 = A + dh * Adot
    B1 = B + dh * Bdot

    mu, Q = eigh(A, B, subset_by_index=(0, m - 1))
    mu1, Q1 = eigh(A1, B1, subset_by_index=(0, m - 1))

    lam, lam1 = mu, mu1

    lam_a, lam_b, N_a, N_b = compute_lam_index(lam, 0, 5)
    eta = softmax_ab(mp.tanh, rho, lam, lam_a, lam_b)
    eta1 = softmax_ab(mp.tanh, rho, lam1, lam_a, lam_b)

    h = fun_h(eta, Q, D)
    h1 = fun_h(eta1, Q1, D)

    ans_cs = (h1 - h) / dh

    ic(np.allclose(A @ Q, B @ Q @ np.diag(lam)))
    ans_approx = deriv_approx(
        A,
        B,
        D,
        Adot,
        Bdot,
        lam,
        Q,
        eta,
        mp.tanh,
        Eij_ab,
        Gij_ab,
        N_a,
        N_b,
        lam_a,
        lam_b,
        rho=rho,
    )

    print("ans_cs =", ans_cs)
    print("ans_approx = ", ans_approx)
    print("error_cs =", np.abs(ans_cs - ans_approx) / np.abs(ans_cs))

    lam, Q = eigh(A, B)
    eta = softmax_ab(mp.tanh, rho, lam, lam_a, lam_b)
    ic(np.allclose(A @ Q, B @ Q @ np.diag(lam)))

    ans = deriv_exict(
        rho,
        A,
        B,
        D,
        Adot,
        Bdot,
        lam,
        Q,
        eta,
        mp.tanh,
        Eij_ab,
        Gij_ab,
        lam_a,
        lam_b,
    )

    print("ans", ans)
    print("error_exict  =", np.abs(ans - ans_approx) / np.abs(ans))

#     plt.plot(lam, eta / np.sum(eta), "o-", label=softmax)
#     plt.legend()
# plt.show()
