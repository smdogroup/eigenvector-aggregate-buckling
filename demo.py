from icecream import ic
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import mpmath as mp
import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import minimize


def rand_symm_mat(n=10, eig_low=0.1, eig_high=100.0, nrepeat=5):
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


def softmax_a(fun, rho, lam, lam_min):
    eta = np.zeros(len(lam))
    for i in range(len(lam)):
        eta[i] = fun(-rho * (lam[i] - lam_min))
    return eta


def softmax_ab(fun, rho, lam, lam_a, lam_b):
    """
    Compute the eta values
    """
    eta = np.zeros(len(lam))
    for i in range(len(lam)):
        a = fun(rho * (lam[i] - lam_a))
        b = fun(rho * (lam[i] - lam_b))
        eta[i] = a - b

    return eta


def compute_lam_ab(lam, a, b):
    """
    Compute the lam_a and lam_b values
    """
    # lam_a = np.argmin(np.abs(lam - a)) - np.min(lam)
    # lam_b = np.argmin(np.abs(lam - b)) + np.min(lam)
    lam_a = np.min(lam[lam > a]) - np.min(np.abs(lam))
    lam_b = np.max(lam[lam < b]) + np.min(np.abs(lam))
    # lam_a = np.min(lam[lam > a])
    # lam_b = np.max(lam[lam < b])
    N_a = np.sum(lam < lam_a)
    N_b = lam.shape[0] - np.sum(lam > lam_b)
    return lam_a, lam_b, N_a, N_b


def fun_h(eta, Q, D, N=None):
    """
    h = sum^N eta_i * q_i^T * D * q_i
    """

    eta = eta / np.sum(eta)

    if N is None:
        N = Q.shape[0]

    h = 0.0
    for i in range(N):
        h += eta[i] * Q[:, i].T @ D @ Q[:, i]

    return h


def fun2_h(rho, D, A, B):
    Binv = np.linalg.inv(B)
    exp = expm(-rho * Binv @ A)
    h = np.trace(exp @ Binv @ D) / np.trace(exp)
    return h


def fun3_h(rho, D, lam, Q):
    """
    h = trace(exp(-rho * lam) * Q^T * D * Q)
    """
    exp = expm(-rho * np.diag(lam))
    exp /= np.trace(exp)
    h = np.trace(exp @ Q.T @ D @ Q)
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
    ndvs=1,
):
    """
    Compute the forward mode derivative
    """

    trace = np.sum(eta)
    eta = eta / trace

    # compute the h value
    h = fun_h(eta, Q, D)
    hdot = np.zeros(ndvs)

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


def deriv_exict2(rho, A, B, D, Adot, Bdot, Ddot=None, ndvs=1):
    lam, Q = eigh(A, B)

    lam_min = np.min(lam)
    eta = np.exp(-rho * (lam - lam_min))
    trace = np.sum(eta)
    eta = eta / trace

    # compute the h value
    h = fun_h(eta, Q, D)
    hdot = np.zeros(ndvs)

    for i in range(np.shape(A)[0]):
        Ar = A - lam[i] * B
        Br = (2 * B @ Q[:, i]).reshape(-1, 1)
        Mat = np.block([[Ar, Br], [-0.5 * Br.T, 0.0]])

        # check if the matrix is singular
        # ic(np.linalg.cond(Mat))

        if np.linalg.cond(Mat) > 1e10:
            ic("Singular matrix")
            continue

        b0 = (-2 * eta[i] * D @ Q[:, i]).reshape(-1, 1)
        b1 = rho * eta[i] * (Q[:, i].T @ D @ Q[:, i] - h)
        b = np.block([[b0], [b1]])

        x = np.linalg.solve(Mat, b)

        pR_px = np.block(
            [
                [((Adot - lam[i] * Bdot) @ Q[:, i]).reshape(-1, 1)],
                [Q[:, i].T @ Bdot @ Q[:, i]],
            ]
        )

        hdot += np.dot(x.T, pR_px).reshape(-1)

        if Ddot is not None:
            hdot += eta[i] * Q[:, i].T @ Ddot @ Q[:, i]

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
    N_a=0,
    N_b=5,
    lam_a=None,
    lam_b=None,
    Ddot=None,
    ndvs=1,
    rho=1.0,
):
    """
    Approximately compute the forward derivative

        first term:
            sum^N E_{ij} (Dq_{ij} + h delta_{ij})*(Aq_dot_{ij} - lam_{i} Bq_dot_{ij})

        second term:
            sum^N 2 * q_{i}^T * Adot * v_{j}
            sum^N q_{i}^T * Bdot * (u_{j} - lam_{j} * v_{j} - w_{j})

    """
    # normalize the eta vector
    trace = np.sum(eta)
    eta = eta / trace

    # compute the h value
    h = fun_h(eta, Q, D)
    hdot = np.zeros(ndvs)

    # only compute the lower triangle of the matrix since it is symmetric
    for i in range(N_a, N_b):
        for j in range(N_a, N_b):
            qDq = Q[:, i].T @ D @ Q[:, j]
            qAdotq = Q[:, i].T @ Adot @ Q[:, j]
            qBdotq = Q[:, i].T @ Bdot @ Q[:, j]

            if Eij_fun == Eij_a:
                Eij = Eij_fun(fun, rho, trace, np.min(lam), lam[i], lam[j])
            else:
                Eij = Eij_fun(fun, rho, trace, lam[i], lam[j], lam_a, lam_b)

            if i == j:
                scale = qDq - h
            else:
                scale = qDq

            hdot += Eij * scale * (qAdotq - lam[j] * qBdotq)

    C = B @ Q[:, N_a:N_b]
    U, _ = np.linalg.qr(C)
    Z = np.eye(np.shape(A)[0]) - U @ U.T

    # compute second term in the derivative approximation
    for j in range(N_a, N_b):
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


def deriv_approx2(
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
    ndvs=1,
    rho=1.0,
):
    # normalize the eta vector
    trace = np.sum(eta)
    eta = eta / trace

    # compute the h value
    h = fun_h(eta, Q, D)
    hdot = np.zeros(ndvs)

    for i in range(N_a, N_b):
        for j in range(N_a, N_b):
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
                scale = qDq

            hdot += scale * (Eij * qAdotq - Gij * qBdotq)

    # compute the orthogonal projector
    C = B @ Q[:, N_a:N_b]
    U, _ = np.linalg.qr(C)
    Z = np.eye(np.shape(A)[0]) - U @ U.T

    for j in range(N_a, N_b):
        Dq = D @ Q[:, j]
        Ak = A - lam[j] * B
        Abar = Z.T @ Ak @ Z
        bbar = Z.T @ (-2.0 * eta[j] * Dq)
        phi = Z @ np.linalg.solve(Abar, bbar)

        hdot += Q[:, j].T @ (Adot - lam[j] * Bdot) @ phi

        hdot -= eta[j] * (Q[:, j].T @ D @ Q[:, j]) * (Q[:, j].T @ Bdot @ Q[:, j])

        if Ddot is not None:
            hdot += eta[j] * Q[:, j].T @ Ddot @ Q[:, j]

    return hdot


# Set parameters
rho = 1000.0
n = 100
dh = 1e-30
ndvs = 5

np.random.seed(123)

x = 0.1 * np.ones(ndvs)
p = np.random.uniform(size=ndvs)

A = rand_symm_mat(n)
B = rand_symm_mat(n)
# B = np.eye(n)
Adot = rand_symm_mat(n)
Bdot = rand_symm_mat(n)
Ddot = rand_symm_mat(n)
D = rand_symm_mat(n)

lam, Q = eigh(A, B)

for softmax in ["exp", "sech", "tanh", "erf", "erfc", "sigmoid", "ncdf"]:
    print("Softmax =", softmax)

    if softmax in ["exp", "sech"]:
        fun = getattr(mp, softmax)

        N = 10
        N_a = 0
        N_b = N
        lam_a = None
        lam_b = None

        eta = softmax_a(fun, rho, lam, np.min(lam))
        Eij_fun = Eij_a
        Gij_fun = Gij_a

    else:
        fun = getattr(mp, softmax)

        a = lam[0] + 10.0
        b = lam[0] + 50.0

        lam_a, lam_b, N_a, N_b = compute_lam_ab(lam, a, b)
        eta = softmax_ab(fun, rho, lam, lam_a, lam_b)

        Eij_fun = Eij_ab
        Gij_fun = Gij_ab

    ans = np.dot(
        deriv_exict(
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
            lam_a,
            lam_b,
            Ddot,
            ndvs=ndvs,
        ),
        p,
    )
    ans_approx = np.dot(
        deriv_approx(
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
            N_a,
            N_b,
            lam_a,
            lam_b,
            Ddot,
            ndvs=ndvs,
            rho=rho,
        ),
        p,
    )
    ans_approx2 = np.dot(
        deriv_approx2(
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
            N_a,
            N_b,
            lam_a,
            lam_b,
            Ddot,
            ndvs=ndvs,
            rho=rho,
        ),
        p,
    )

    # print("ans = ", ans)
    # print("ans_approx1 = ", ans_approx)
    # print("ans_approx2 = ", ans_approx2)
    print("error1  =", np.abs(ans - ans_approx) / np.abs(ans))
    print("error2  =", np.abs(ans - ans_approx2) / np.abs(ans))
    print("")

    plt.plot(lam, eta, "o-", label=softmax)
    plt.legend()
plt.show()
