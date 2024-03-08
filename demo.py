from icecream import ic
import matplotlib.pylab as plt
import mpmath as mp
import numpy as np
from scipy.linalg import eigh, eigvalsh, expm
import scipy.sparse


def rand_symm_mat(n=10, eig_low=0.1, eig_high=100.0, nrepeat=3):
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


def softmax_a(fun, rho, lam):
    eta = np.zeros(len(lam), dtype=lam.dtype)
    for i in range(len(lam)):
        eta[i] = fun(-rho * (lam[i] - np.min(lam)))
    return eta


def softmax_ab(fun, rho, lam, lam_a, lam_b):
    """
    Compute the eta values
    """
    # ic(rho, lam, lam_a, lam_b)
    eta = np.zeros(len(lam), dtype=lam.dtype)
    for i in range(len(lam)):
        a = fun(rho * (lam[i] - lam_a))
        b = fun(rho * (lam[i] - lam_b))
        eta[i] = a - b
    # ic(eta[:10])
    return eta


def compute_lam_value(lam, a, b):
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


# compute lam_a and lam_b based on the index
def compute_lam_index(lam, N_a, N_b):
    lam_a = lam[N_a] - np.min(np.abs(lam)) * 0.1
    lam_b = lam[N_b] + np.min(np.abs(lam)) * 0.1
    N_b += 1
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


def Fij_a(fun, rho, trace, lam_min, lam1, lam2):
    with mp.workdps(80):
        if lam1 == lam2:
            val = (
                -rho * lam1 * fun(-rho * (lam1 - lam_min)) / trace
                + fun(-rho * (lam1 - lam_min)) / trace
            )
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
        else:
            val = (eta1 - eta2) / (mp.mpf(lam1) - mp.mpf(lam2)) / mp.mpf(trace)
    return np.float64(val)


def Fij_ab(fun, rho, trace, lam1, lam2, lam_a, lam_b):
    with mp.workdps(80):
        a1 = fun(rho * (mp.mpf(lam1) - mp.mpf(lam_a)))
        b1 = fun(rho * (mp.mpf(lam1) - mp.mpf(lam_b)))
        a2 = fun(rho * (mp.mpf(lam2) - mp.mpf(lam_a)))
        b2 = fun(rho * (mp.mpf(lam2) - mp.mpf(lam_b)))

        eta1 = a1 - b1
        eta2 = a2 - b2

        if lam1 == lam2:
            val = -rho * lam2 * eta2 * (a2 + b2) / mp.mpf(trace) + eta1 / mp.mpf(trace)
        else:
            val = (
                (lam1 * eta1 - lam2 * eta2)
                / (mp.mpf(lam1) - mp.mpf(lam2))
                / mp.mpf(trace)
            )
    return np.float64(val)


def deriv_exict_EF(
    rho,
    A,
    D,
    Adot,
    Bdot,
    lam,
    Q,
    eta,
    fun,
    Eij_fun,
    Fij_fun,
    lam_a=None,
    lam_b=None,
    Ddot=None,
):
    """
    Compute the forward mode derivative
    """

    trace = np.sum(eta)
    eta = eta / trace
    h = fun_h(eta, Q, D)
    hdot = 0.0

    for j in range(np.shape(A)[0]):
        for i in range(np.shape(A)[0]):
            Adot_q = Q[:, i].T @ Adot @ Q[:, j]
            Bdot_q = Q[:, i].T @ Bdot @ Q[:, j]
            qDq = Q[:, i].T @ D @ Q[:, j]

            if Eij_fun == Eij_a:
                Eij = Eij_fun(fun, rho, trace, np.min(lam), lam[i], lam[j])
                Fij = Fij_fun(fun, rho, trace, np.min(lam), lam[i], lam[j])
            else:
                Eij = Eij_fun(fun, rho, trace, lam[i], lam[j], lam_a, lam_b)
                Fij = Fij_fun(fun, rho, trace, lam[i], lam[j], lam_a, lam_b)

            if i == j:
                scale = qDq - h
            else:
                scale = qDq

            hdot += scale * (Eij * Adot_q - Fij * Bdot_q)

    for j in range(np.shape(A)[0]):
        hdot -= eta[j] * h * (Q[:, j].T @ Bdot @ Q[:, j])

        if Ddot is not None:
            hdot += eta[j] * Q[:, j].T @ Ddot @ Q[:, j]

    return hdot


def deriv_exict_E(
    rho,
    A,
    D,
    Adot,
    Bdot,
    lam,
    Q,
    eta,
    fun,
    Eij_fun,
    lam_a=None,
    lam_b=None,
    Ddot=None,
):
    """
    Compute the forward mode derivative
    """

    trace = np.sum(eta)
    eta = eta / trace
    h = fun_h(eta, Q, D)
    hdot = 0.0

    for j in range(np.shape(A)[0]):
        for i in range(np.shape(A)[0]):
            Adot_q = Q[:, i].T @ Adot @ Q[:, j]
            Bdot_q = Q[:, i].T @ Bdot @ Q[:, j]
            Dq = Q[:, i].T @ D @ Q[:, j]

            if Eij_fun == Eij_a:
                Eij = Eij_fun(fun, rho, trace, np.min(lam), lam[i], lam[j])
            else:
                Eij = Eij_fun(fun, rho, trace, lam[i], lam[j], lam_a, lam_b)

            if i == j:
                hdot += Eij * (Dq - h) * (Adot_q - lam[j] * Bdot_q)
            else:
                hdot += Eij * Dq * (Adot_q - lam[j] * Bdot_q)

            hdot -= eta[i] * Dq * Bdot_q

    for j in range(np.shape(A)[0]):
        if Ddot is not None:
            hdot += eta[j] * Q[:, j].T @ Ddot @ Q[:, j]

    return hdot


def deriv_exict3(rho, A, B, D, Adot, Bdot, Ddot=None):
    lam, Q = eigh(A, B)

    lam_min = np.min(lam)
    eta = np.exp(-rho * (lam - lam_min))
    trace = np.sum(eta)
    eta = eta / trace

    # compute the h value
    h = fun_h(eta, Q, D)
    hdot = 0.0

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


def deriv_approx_EF(
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
    Fij_fun,
    N_a=0,
    N_b=5,
    lam_a=None,
    lam_b=None,
    Ddot=None,
    rho=1.0,
):
    """
    Eignvector Aggregation II
    """
    # normalize the eta vector
    trace = np.sum(eta)
    eta = eta / trace
    h = fun_h(eta, Q, D)
    hdot = 0.0

    for i in range(N_a, N_b):
        for j in range(i + 1):
            qDq = Q[:, i].T @ D @ Q[:, j]
            qAdotq = Q[:, i].T @ Adot @ Q[:, j]
            qBdotq = Q[:, i].T @ Bdot @ Q[:, j]

            if Eij_fun == Eij_a:
                Eij = Eij_fun(fun, rho, trace, np.min(lam), lam[i], lam[j])
                Fij = Fij_fun(fun, rho, trace, np.min(lam), lam[i], lam[j])
            else:
                Eij = Eij_fun(fun, rho, trace, lam[i], lam[j], lam_a, lam_b)
                Fij = Fij_fun(fun, rho, trace, lam[i], lam[j], lam_a, lam_b)

            if i == j:
                scale = qDq - h
            else:
                scale = 2 * qDq

            hdot += scale * (Eij * qAdotq - Fij * qBdotq)

    # compute the orthogonal projector
    C = B @ Q[:, :N_b]
    U, _ = np.linalg.qr(C)

    factor = 0.99
    P = A - factor * lam[0] * B
    P = scipy.sparse.csc_matrix(P)
    Pfactor = scipy.sparse.linalg.factorized(P)

    def preconditioner(x):
        y = Pfactor(x)
        t = np.dot(U.T, y)
        y = y - np.dot(U, t)
        return y

    preop = scipy.sparse.linalg.LinearOperator((n, n), matvec=preconditioner)

    for j in range(np.max([0, N_a - 1]), N_b):
        Dq = D @ Q[:, j]
        bkr = -2.0 * Dq

        def matrix(x):
            y = A.dot(x) - lam[j] * B.dot(x)
            t = np.dot(U.T, y)
            y = y - np.dot(U, t)
            return y

        matop = scipy.sparse.linalg.LinearOperator((n, n), matvec=matrix)

        t = np.dot(U.T, bkr)
        bkr = bkr - np.dot(U, t)
        phi, _ = scipy.sparse.linalg.gmres(matop, bkr, M=preop, rtol=1e-10, atol=1e-15)

        hdot += eta[j] * Q[:, j].T @ (Adot - lam[j] * Bdot) @ phi
        hdot -= eta[j] * h * (Q[:, j].T @ Bdot @ Q[:, j])

        if Ddot is not None:
            hdot += eta[j] * Q[:, j].T @ Ddot @ Q[:, j]

    return hdot


def deriv_approx_E(
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
    rho=1.0,
):
    """
    Eignvector Aggregation I
    """
    # normalize the eta vector
    trace = np.sum(eta)
    eta = eta / trace
    h = fun_h(eta, Q, D)
    hdot = 0.0

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


def store_EG(fun, Gr, Kr, ks_rho, N_a, N_b):
    n = 100

    mu = eigvalsh(Gr.todense(), Kr.todense(), eigvals=(0, n))
    lam_a, lam_b = mu[N_a], mu[N_b]
    eta = softmax_ab(fun, ks_rho, mu, lam_a, lam_b)
    eta_sum = np.sum(eta)

    ic(mu[:20])
    ic(eta[:20] / eta_sum)

    E = np.zeros((n, n))
    G = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            E[i, j] = Eij_ab(fun, ks_rho, eta_sum, mu[i], mu[j], lam_a, lam_b)
            G[i, j] = Fij_ab(fun, ks_rho, eta_sum, mu[i], mu[j], lam_a, lam_b)

    np.save("data/E.npy", E)
    np.save("data/G.npy", G)


if __name__ == "__main__":
    # Set parameters
    rho = 1000.0  # how large rho is depends on how close the eigenvalues are
    n = 100
    dh = 1e-8
    dhh = 1e-50
    ndvs = 1

    np.random.seed(0)

    A = rand_symm_mat(n)
    # B = rand_symm_mat(n)
    B = np.eye(n)
    D = rand_symm_mat(n)

    Adot = rand_symm_mat(n)
    Bdot = rand_symm_mat(n)
    Ddot = rand_symm_mat(n)

    # use central difference to check the derivative
    A1 = A + dh * Adot
    B1 = B + dh * Bdot
    D1 = D + dh * Ddot

    A2 = A - dh * Adot
    B2 = B - dh * Bdot
    D2 = D - dh * Ddot

    lam, Q = eigh(A, B)
    lam1, Q1 = eigh(A1, B1)
    lam2, Q2 = eigh(A2, B2)

    # ["exp", "sech", "tanh", "erf", "erfc", "sigmoid", "ncdf"]:
    for softmax in ["tanh", "exp"]:
        print("Softmax =", softmax)

        if softmax in ["exp"]:
            fun = getattr(mp, softmax)

            N = 10
            N_a = 0
            N_b = N
            lam_a = None
            lam_b = None

            eta = softmax_a(fun, rho, lam)
            eta1 = softmax_a(fun, rho, lam1)
            eta2 = softmax_a(fun, rho, lam2)

            Eij_fun = Eij_a
            Fij_fun = Fij_a

            h1 = fun_h(eta1, Q1, D1)
            h2 = fun_h(eta2, Q2, D2)

            ans_cs = (h1 - h2) / (2 * dh)

        else:
            fun = getattr(mp, softmax)

            # compute the lam_a and lam_b based on the value
            # a = lam[0] + 10.0
            # b = lam[0] + 50.0
            # lam_a, lam_b, N_a, N_b = compute_lam_value(lam, a, b)

            # compute the lam_a and lam_b based on the index
            lam_a, lam_b, N_a, N_b = compute_lam_index(lam, 0, 6)

            eta = softmax_ab(fun, rho, lam, lam_a, lam_b)
            eta1 = softmax_ab(fun, rho, lam1, lam_a, lam_b)
            eta2 = softmax_ab(fun, rho, lam2, lam_a, lam_b)

            h1 = fun_h(eta1, Q1, D1)
            h2 = fun_h(eta2, Q2, D2)

            ans_cs = (h1 - h2) / (2 * dh)

            Eij_fun = Eij_ab
            Fij_fun = Fij_ab

        ans_exict = deriv_exict_EF(
            rho,
            A,
            D,
            Adot,
            Bdot,
            lam,
            Q,
            eta,
            fun,
            Eij_fun,
            Fij_fun,
            lam_a,
            lam_b,
            Ddot,
        )

        ans_approx = deriv_approx_EF(
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
            Fij_fun,
            N_a,
            N_b,
            lam_a,
            lam_b,
            Ddot,
            rho=rho,
        )

        print("ans_exict", ans_exict)
        print("ans_cs =", ans_cs)
        print("cs_exict =", np.abs(ans_exict - ans_cs) / np.abs(ans_cs))
        print(
            "exict_approx =", np.abs(ans_exict - ans_approx) / np.abs(ans_exict), "\n"
        )
