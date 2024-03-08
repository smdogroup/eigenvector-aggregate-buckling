import time

from icecream import ic
import numpy as np
from scipy import sparse, spatial
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, linalg
from scipy.sparse.linalg import eigs, eigsh, inv

import kokkos as kokkos


def lobpcg4(A, X, B=None, M=None, tol=1e-8, maxiter=500):
    N, m0 = X.shape
    m1 = int(np.ceil(3.0 * m0))
    m = m0 + m1
    ic(m0, m1)

    if B is None:
        B = np.eye(N)
    # X = np.eye(N, m) / np.sqrt(N)
    X = np.eye(N, m)
    XAX = A[:m, :m]
    XBX = B[:m, :m]
    # C, O = RayleighRitz3(XAX, XBX, m)
    O, C = eigh(XAX, XBX)

    X = X @ C
    # X = X / np.linalg.norm(X)
    R = A @ X - B @ X @ np.diag(O)
    P = np.zeros((N, m))

    non_convergent_indx = np.arange(m)

    AX = A @ X
    AW = A @ R
    AP = A @ P
    BX = B @ X
    BW = B @ R
    BP = B @ P

    for k in range(maxiter):
        if M is None:
            W = R
        else:
            W = M @ R
            
        AW = A @ W
        BW = B @ W
        
        # if k == 0:
        #     ic(W[0, :])
        #     ic(R[0, :])
        #     ic(AP[0, :])
        #     ic(BP[0, :])
        #     ic(AW[0, :])
        #     ic(BW[0, :])

        for i in non_convergent_indx:
            # Xi_norm = np.linalg.norm(X[:, i])
            # Wi_norm = np.linalg.norm(W[:, i])
            # Xi_norm = np.max(np.abs(X[:, i]))
            # Wi_norm = np.max(np.abs(W[:, i]))

            # W[:, i] = W[:, i] / Wi_norm
            # X[:, i] = X[:, i] / Xi_norm

            # AX[:, i] = AX[:, i] / Xi_norm
            # AW[:, i] = AW[:, i] / Wi_norm

            # BX[:, i] = BX[:, i] / Xi_norm
            # BW[:, i] = BW[:, i] / Wi_norm

            # if k > 0:
            #     Pi_norm = np.linalg.norm(P[:, i])
            #     Pi_norm = np.max(np.abs(P[:, i]))
            #     P[:, i] = P[:, i] / Pi_norm
            #     AP[:, i] = AP[:, i] / Pi_norm
            #     BP[:, i] = BP[:, i] / Pi_norm
                
            S = np.vstack((X[:, i], W[:, i], P[:, i])).T
            AS = np.vstack((AX[:, i], AW[:, i], AP[:, i])).T
            BS = np.vstack((BX[:, i], BW[:, i], BP[:, i])).T

            # S = S / np.linalg.norm(S, axis=0)

            SAS = S.T @ AS
            SBS = S.T @ BS
            
            # ic(SAS)
            # ic(SBS)

            if k > 0:
                O, C = eigh(SAS, SBS, subset_by_index=[0, 0])
                # Oa, Ca = sygvx3x3(SAS, SBS)
            else:
                O, C = eigh(SAS[:2, :2], SBS[:2, :2], subset_by_index=[0, 0])
                # Oa, Ca = sygvx2x2(SAS[:2, :2], SBS[:2, :2])

            # ic(np.allclose(np.abs(C[:, 0]), np.abs(Ca[:, 0]), atol=1e-8))

            # O = Oa[0]
            # C = Ca
            
            # ic(O)
            # ic(C)

            if k > 0:
                P[:, i] = C[1, 0] * W[:, i] + C[2, 0] * P[:, i]
                AP[:, i] = C[1, 0] * AW[:, i] + C[2, 0] * AP[:, i]
                BP[:, i] = C[1, 0] * BW[:, i] + C[2, 0] * BP[:, i]
            else:
                P[:, i] = C[1, 0] * W[:, i]
                AP[:, i] = C[1, 0] * AW[:, i]
                BP[:, i] = C[1, 0] * BW[:, i]

            X[:, i] = P[:, i] + C[0, 0] * X[:, i]
            AX[:, i] = AP[:, i] + C[0, 0] * AX[:, i]
            BX[:, i] = BP[:, i] + C[0, 0] * BX[:, i]

        XAX = X.T @ AX
        XBX = X.T @ BX
        O, C = eigh(XAX, XBX)

        X = X @ C
        P = P @ C
        AX = AX @ C
        BX = BX @ C
        AP = AP @ C
        BP = BP @ C
        R = BX @ np.diag(O) - AX

        res_max = 0
        non_convergent_indx = []

        for i in range(m0):
            # residual = np.linalg.norm(R[:, i]) / (
            #     (Anorm + Bnorm * O[i]) * np.linalg.norm(X[:, i])
            # )
            residual = np.linalg.norm(R[:, i]) / np.linalg.norm(
                AX[:, i] + BX[:, i] * O[i]
            )
            # ic(residual)
            if residual > res_max:
                res_max = residual
            if residual > tol:
                non_convergent_indx.append(i)

        counter = m0 - len(non_convergent_indx)
        ic(k, counter, res_max)

        if res_max < tol:
            break

        for i in range(m0, m):
            # residual = np.linalg.norm(R[:, i]) / (
            #     (Anorm + Bnorm * np.abs(O[i])) * np.linalg.norm(X[:, i])
            # )
            residual = np.linalg.norm(R[:, i]) / np.linalg.norm(
                AX[:, i] + BX[:, i] * O[i]
            )
            if residual > tol:
                non_convergent_indx.append(i)
    # ic(X[:, :m0])
    return O[:m0], X[:, :m0]

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

    return np.dot(Q, np.dot(np.diag(lam), Q.T))  # Compute G = Q*Lambda*Q^{T}


if __name__ == "__main__":
    n = 1000  #  1000000
    m = 2  # Number of desired eigenpairs
    nnz = 17 * n  # 17964016
    dens = nnz / (n * n)
    np.random.seed(0)

    # G = np.random.rand(n, n)
    # G = G + G.T
    # # G = G + n * np.eye(n)

    # K = np.random.rand(n, n)
    # K = K + K.T
    # K = K + n * np.eye(n)
    # G = sparse.csr_matrix(G)
    # K = sparse.csr_matrix(K)

    # generate symmetric sparse matrix
    
    G = sparse.random(n, n, density=0.1, format="csr", dtype=np.float64)
    G = G + G.T
    # G = G + n * sparse.eye(n, dtype=np.float64)

    K = sparse.random(n, n, density=0.1, format="csr", dtype=np.float64)
    K = K + K.T
    K = K + n * sparse.eye(n, dtype=np.float64)
    
    # ic(len(G.data), len(G.indices), len(G.indptr))
    # ic(len(K.data), len(K.indices), len(K.indptr))
    # ic(G.data, G.indices, G.indptr)
    # ic(K.data, K.indices, K.indptr)

    # ic(G.todense())
    # ic(K.todense())

    # ic(np.reshape(G.todense(), (n * n)))
    # ic(np.reshape(K.todense(), (n * n)))

    # start = time.time()
    # mu, Qr = eigh(G.todense(), K.todense())
    # end = time.time()
    # print("scipy::eigh: ", end - start)
    # eigs = -1.0 / mu
    # ic(eigs[:m])
    
    print("scipy::eigsh")
    start = time.time()
    mu, Qr = eigsh(G, M=K, k=m, which="SM", sigma=10000.0)
    end = time.time()
    print("scipy::eigsh: ", end - start)
    eigs = -1.0 / mu
    ic(eigs)

    kokkos.initialize()
    print("kokkos::lobpcg")
    start = time.time()
    mu, Qr = kokkos.lobpcg(
        G.data, G.indptr, G.indices, K.data, K.indptr, K.indices, n, m, np.linalg.inv(K.todense())
    )
    end = time.time()
    kokkos.finalize()
    print("kokkos::lobpcg: ", end - start)

    A = np.array(G.todense())
    B = np.array(K.todense())
    M = np.linalg.inv(B)
    X = np.eye(n, m)
    mu, Qr = lobpcg4(A, X, B, M=M)

    eigs = -1.0 / mu
    ic(eigs)


