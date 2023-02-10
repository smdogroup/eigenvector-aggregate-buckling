import numpy as np
from scipy.linalg import eigh, expm
import matplotlib.pylab as plt
import mpmath as mp


class EulerBeam:
    """
    Optimization code for a clamped-clamped beam
    """

    def __init__(self, L, nelems, ndvs=5, E=1.0, density=1.0):
        self.L = L
        self.nelems = nelems
        self.ndof = 2 * (self.nelems - 1)
        self.ndvs = ndvs
        self.E = E
        self.density = density

        self.ksrho = 100.0
        self.D = np.zeros((self.ndof, self.ndof))

        u = np.linspace(0.5 / self.nelems, 1.0 - 0.5 / self.nelems, self.nelems)
        self.N = self.eval_bernstein(u, self.ndvs)

        # Length of one element
        Le = self.L / self.nelems

        # Compute the stiffness matrix
        self.ke = np.array(
            [
                [12.0, 6.0 * Le, -12.0, 6.0 * Le],
                [6.0 * Le, 4.0 * Le**2, -6.0 * Le, 2.0 * Le**2],
                [-12.0, -6.0 * Le, 12.0, -6.0 * Le],
                [6.0 * Le, 2.0 * Le**2, -6.0 * Le, 4.0 * Le**2],
            ]
        )
        self.ke *= 1.0 / Le**3

        # Compute the mass matrix
        self.me = np.array(
            [
                [156.0, 22.0 * Le, 54.0, -13.0 * Le],
                [22.0 * Le, 4 * Le**2, 13.0 * Le, -3.0 * Le**2],
                [54.0, 13.0 * Le, 156.0, -22.0 * Le],
                [-13.0 * Le, -3.0 * Le**2, -22.0 * Le, 4.0 * Le**2],
            ]
        )
        self.me *= Le / 420.0

        return

    def eval_bernstein(self, u, order):
        """
        Evaluate the interpolation between the design variables and the stiffness or mass distribution
        """
        u1 = 1.0 - u
        u2 = 1.0 * u

        N = np.zeros((len(u), order))
        N[:, 0] = 1.0

        for j in range(1, order):
            s = np.zeros(len(u))
            t = np.zeros(len(u))
            for k in range(j):
                t[:] = N[:, k]
                N[:, k] = s + u1 * t
                s = u2 * t
            N[:, j] = s

        return N

    def get_vars(self, elem):
        """
        Get the global variables for the given element
        """
        if elem == 0:
            return [-1, -1, 0, 1]
        elif elem == self.nelems - 1:
            return [2 * (elem - 1), 2 * (elem - 1) + 1, -1, -1]
        else:
            return [2 * (elem - 1), 2 * (elem - 1) + 1, 2 * elem, 2 * elem + 1]

    def mass_matrix(self, x):
        """
        Compute the mass matrix for the clamped-clamped beam
        """

        rhoA = self.density * np.dot(self.N, x)

        M = np.zeros((self.ndof, self.ndof), dtype=rhoA.dtype)
        for k in range(self.nelems):
            elem_vars = self.get_vars(k)
            for ie, i in enumerate(elem_vars):
                for je, j in enumerate(elem_vars):
                    if i >= 0 and j >= 0:
                        M[i, j] += rhoA[k] * self.me[ie, je]

        return M

    def mass_matrix_deriv(self, u, v):
        dfdrhoA = np.zeros((self.nelems))

        for k in range(self.nelems):
            elem_vars = self.get_vars(k)
            for ie, i in enumerate(elem_vars):
                for je, j in enumerate(elem_vars):
                    if i >= 0 and j >= 0:
                        dfdrhoA[k] += u[i] * v[j] * self.me[ie, je]

        return self.density * np.dot(self.N.T, dfdrhoA)

    def stiffness_matrix(self, x):
        """
        Compute the stiffness matrix of the clamped-clamped beam
        """

        EI = self.E * np.dot(self.N, x)

        K = np.zeros((self.ndof, self.ndof), dtype=EI.dtype)
        for k in range(self.nelems):
            elem_vars = self.get_vars(k)
            for ie, i in enumerate(elem_vars):
                for je, j in enumerate(elem_vars):
                    if i >= 0 and j >= 0:
                        K[i, j] += EI[k] * self.ke[ie, je]

        return K

    def stiffness_matrix_deriv(self, u, v):
        dfdEI = np.zeros((self.nelems))
        for k in range(self.nelems):
            elem_vars = self.get_vars(k)
            for ie, i in enumerate(elem_vars):
                for je, j in enumerate(elem_vars):
                    if i >= 0 and j >= 0:
                        dfdEI[k] += u[i] * v[j] * self.ke[ie, je]

        return self.E * np.dot(self.N.T, dfdEI)

    def appox_min_eigenvalue(self, x):

        A = self.stiffness_matrix(x)
        B = self.mass_matrix(x)

        # Compute the eigenvalues of the generalized eigen problem
        lam, Q = eigh(A, B)

        min_lam = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - min_lam))
        return min_lam - np.log(np.sum(eta)) / self.ksrho

    def approx_min_eigenvalue_deriv(self, x, N=5):
        A = self.stiffness_matrix(x)
        B = self.mass_matrix(x)

        # Compute the eigenvalues of the generalized eigen problem
        lam, Q = eigh(A, B)

        min_lam = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - min_lam))
        eta = eta / np.sum(eta)
        dfdx = np.zeros(self.ndvs)

        for k in range(N):
            dfdx += eta[k] * (
                self.stiffness_matrix_deriv(Q[:, k], Q[:, k])
                - lam[k] * self.mass_matrix_deriv(Q[:, k], Q[:, k])
            )

        return dfdx

    def exact_eigvector(self, x):
        """
        Compute the eigenvector constraint

        h = tr(D * B^{-1} * exp(- rho * A * B^{-1})/ tr(exp(- rho * A * B^{-1}))
        """

        A = self.stiffness_matrix(x)
        B = self.mass_matrix(x)

        Binv = np.linalg.inv(B)
        exp = expm(-self.ksrho * np.dot(A, Binv))
        h = np.trace(np.dot(self.D, np.dot(Binv, exp))) / np.trace(exp)

        return h

    def approx_eigenvector(self, x, N=5):

        A = self.stiffness_matrix(x)
        B = self.mass_matrix(x)

        # Compute the eigenvalues of the generalized eigen problem
        lam, Q = eigh(A, B)

        eta = np.exp(-self.ksrho * (lam - np.min(lam)))
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(N):
            h += eta[i] * np.dot(Q[:, i], np.dot(self.D, Q[:, i]))

        return h

    def plot_modes(self, x, N=5):

        u = np.linspace(0.5 / self.nelems, 1.0 - 0.5 / self.nelems, self.nelems)
        xvals = np.dot(self.N, x)

        plt.figure()
        plt.plot(u, xvals)

        A = self.stiffness_matrix(x)
        B = self.mass_matrix(x)
        lam, Q = eigh(A, B)

        plt.figure()
        for k in range(5):
            u = np.zeros(self.nelems + 1)
            x = np.linspace(0, self.L, self.nelems + 1)
            u[1:-1] = Q[::2, k]
            plt.plot(x, u)

        plt.show()

    def precise(self, rho, trace, lam_min, lam1, lam2):
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

    def exact_eigvector_deriv(self, x):
        """
        Compute the exact derivative
        """

        A = self.stiffness_matrix(x)
        B = self.mass_matrix(x)

        # Compute the eigenvalues of the generalized eigen problem
        lam, Q = eigh(A, B)

        lam_min = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - lam_min))
        trace = np.sum(eta)
        eta = eta / trace

        h = 0.0
        for i in range(A.shape[0]):
            h += eta[i] * np.dot(Q[:, i], np.dot(self.D, Q[:, i]))

        dfdx = np.zeros(self.ndvs)

        G = np.dot(np.diag(eta), np.dot(Q.T, np.dot(self.D, Q)))
        for j in range(A.shape[1]):
            for i in range(A.shape[0]):
                qDq = np.dot(Q[:, i], np.dot(self.D, Q[:, j]))
                scalar = qDq
                if i == j:
                    scalar = qDq - h

                Eij = scalar * self.precise(self.ksrho, trace, lam_min, lam[i], lam[j])

                # Add to dfdx from A
                dfdx += Eij * self.stiffness_matrix_deriv(Q[:, i], Q[:, j])

                # Add to dfdx from B
                dfdx -= (Eij * lam[j] + G[i, j]) * self.mass_matrix_deriv(
                    Q[:, i], Q[:, j]
                )

        return dfdx

    def approx_eigenvector_deriv(self, x, N=5):
        """
        Approximately compute the forward derivative
        """

        # Compute the mass and stiffness matrices
        A = self.stiffness_matrix(x)
        B = self.mass_matrix(x)

        # Compute the eigenvalues of the generalized eigen problem
        lam, Q = eigh(A, B)

        # Take only the smallest N eigenvalues
        QN = Q[:, :N]

        lam_min = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - lam_min))
        trace = np.sum(eta)
        eta = eta / trace

        # Compute the h values
        h = 0.0
        for i in range(N):
            h += eta[i] * np.dot(QN[:, i], np.dot(self.D, QN[:, i]))

        # Set the value of the derivative
        dfdx = np.zeros(self.ndvs)

        for j in range(N):
            for i in range(N):
                qDq = np.dot(QN[:, i], np.dot(self.D, QN[:, j]))
                scalar = qDq
                if i == j:
                    scalar = qDq - h

                Eij = scalar * self.precise(self.ksrho, trace, lam_min, lam[i], lam[j])

                # Add to dfdx from A
                dfdx += Eij * self.stiffness_matrix_deriv(QN[:, i], QN[:, j])

                # Add to dfdx from B
                dfdx -= Eij * lam[i] * self.mass_matrix_deriv(QN[:, i], QN[:, j])

        # Form the augmented linear system of equations
        for k in range(N):
            # Compute B * vk = D * qk
            vk = np.linalg.solve(B, -eta[k] * np.dot(self.D, QN[:, k]))
            dfdx += self.mass_matrix_deriv(QN[:, k], vk)

            # Solve the augmented system of equations for wk
            Ak = A - lam[k] * B
            Ck = np.dot(B, QN)

            # Set up the augmented linear system of equations
            mat = np.block([[Ak, Ck], [Ck.T, np.zeros((N, N))]])
            b = np.zeros(mat.shape[0])

            # Compute the right-hand-side vector
            bk = np.dot(self.D, QN[:, k])
            b[: self.ndof] = -eta[k] * bk

            # Solve the first block linear system of equations
            sol = np.linalg.solve(mat, b)
            wk = sol[: self.ndof]

            # Compute the contributions from the derivative from Adot
            dfdx += 2.0 * self.stiffness_matrix_deriv(QN[:, k], wk)

            # Add the contributions to the derivative from Bdot here...
            dfdx -= lam[k] * self.mass_matrix_deriv(QN[:, k], wk)

            # Now, compute the remaining contributions to the derivative from
            # B by solving the second auxiliary system
            dk = np.dot(A, vk)
            b[: self.ndof] = dk

            sol = np.linalg.solve(mat, b)
            uk = sol[: self.ndof]

            # Compute the contributions from the derivative
            dfdx -= self.mass_matrix_deriv(QN[:, k], uk)

        return dfdx


np.random.seed(12345)


ndvs = 10
L = 1.0
nelems = 51
beam = EulerBeam(L, nelems, ndvs=ndvs)

# Set the matrix component we want to zero
dof = np.arange(0, nelems // 4)
beam.D[dof, dof] = 1.0

x0 = 0.01 * np.ones(ndvs)

from scipy.optimize import minimize

obj = lambda x: 0.1 * beam.approx_eigenvector(x)  # >= 0.0
obj_grad = lambda x: 0.1 * beam.approx_eigenvector_deriv(x)

# obj = lambda x: -0.01 * beam.appox_min_eigenvalue(x)
# obj_grad = lambda x: -0.01 * beam.approx_min_eigenvalue_deriv(x)

con_mass = lambda x: 0.5 * beam.nelems - np.sum(np.dot(beam.N, x))
con_mass_grad = lambda x: -np.sum(beam.N, axis=0)

x = np.random.uniform(size=ndvs)
px = np.random.uniform(size=ndvs)

# dh = 1e-5
# gx = con_mass_grad(x)
# fd = (con_mass(x + dh * px) - con_mass(x)) / dh
# ans = np.dot(gx, px)
# print("fd  = ", fd)
# print("ans = ", ans)

res = minimize(
    obj,
    x0,
    jac=obj_grad,
    method="SLSQP",
    bounds=[(1e-3, 10.0)] * len(x0),
    constraints=[
        {"type": "ineq", "fun": con_mass, "jac": con_mass_grad},
        # {
        #     "type": "eq",
        #     "fun": con_mass,
        #     "jac": con_mass_grad,
        # },
    ],
)

print(res)


# A = beam.stiffness_matrix(x)
# B = beam.mass_matrix(x)
# lam, Q = eigh(A, B)
# print(np.dot(Q[:, 0], np.dot(beam.D, Q[:, 0])))
# print(beam.approx_eigenvector(x))

# print("min(lam) = ", min(lam))
# print(beam.appox_min_eigenvalue(x))

# # Random search direction
# px = np.random.uniform(size=x.shape)

# dh = 1e-5
# fd = (beam.approx_eigenvector(x + dh * px) - beam.approx_eigenvector(x)) / dh

# # dh = 1e-30
# # fd = beam.exact_eigvector(x + 1j * dh * px).imag / dh
# # ans_exact = np.dot(px, beam.exact_eigvector_deriv(x))
# ans_approx = np.dot(px, beam.approx_eigenvector_deriv(x, N=5))

# print("fd = ", fd)
# # print("ans_exact = ", ans_exact)
# print("ans_approx = ", ans_approx)
beam.plot_modes(res.x)


# fd = (beam.appox_min_eigenvalue(x + dh * px) - beam.appox_min_eigenvalue(x)) / dh
# ans_approx = np.dot(px, beam.approx_min_eigenvalue_deriv(x))

# print("fd = ", fd)
# print("ans_approx = ", ans_approx)
