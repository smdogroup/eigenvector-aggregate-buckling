from icecream import ic
import matplotlib as mpl
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D, proj3d
import mpmath as mp
import numpy as np
import scienceplots
from scipy import interpolate
from scipy.linalg import eigh, expm
from scipy.optimize import minimize


class EulerBeam:
    """
    Optimization code for a clamped-clamped beam
    """

    def __init__(
        self, nelems, ndvs=5, L=1.0, t=0.01, N=5, ksrho=10.0, E=1.0, density=1.0
    ):
        """
        Initizlie the data for the clamped-clamped beam eigenvalue problem.
        Parameters
        ----------
        L : float
            Length of the beam
        nelems : int
            Number of elements along the length of the beam
        ndvs : int
            Number of design variables
        E : float
            Elastic modulus
        density : float
            Density of the beam
        t : float
            Thickness of the tube
        N : int
            Number of eigenvalues to compute to do the approximation
        ksrho : float
            Approximation parameter for eta
        """

        self.L = L
        self.nelems = nelems
        self.ndof = 4 * (self.nelems - 1)
        self.ndvs = ndvs
        self.E = E
        self.density = density
        self.t = t

        self.Np = N
        self.ksrho = ksrho
        self.D = np.zeros((self.ndof, self.ndof))

        u = np.linspace(0.5 / self.nelems, 1.0 - 0.5 / self.nelems, self.nelems)
        self.N = self.eval_bernstein(u, self.ndvs)

        # Length of one element
        Le = self.L / self.nelems

        # dof are stored by: v, w, theta,y, theta,z
        # v, theta,z - beam deformation in the y-z plane
        xydof = [0, 3, 4, 7]
        # w, theta,y - beam deformation in the x-z plane
        xzdof = [1, 2, 5, 6]

        # Set the transformations for the x-z plane and the x-y plane
        cxy = np.array([1.0, Le, 1.0, Le])
        cxz = np.array([1.0, -Le, 1.0, -Le])

        # Compute the stiffness matrix
        k0 = np.array(
            [
                [12.0, -6.0, -12.0, -6.0],
                [-6.0, 4.0, 6.0, 2.0],
                [-12.0, 6.0, 12.0, 6.0],
                [-6.0, 2.0, 6.0, 4.0],
            ]
        )
        ky = (k0 / Le**3) * np.outer(cxy, cxy)
        kz = (k0 / Le**3) * np.outer(cxz, cxz)

        # Set the elements into the stiffness matrix
        self.ke = np.zeros((8, 8))
        for ie, i in enumerate(xydof):
            for je, j in enumerate(xydof):
                self.ke[i, j] = ky[ie, je]

        for ie, i in enumerate(xzdof):
            for je, j in enumerate(xzdof):
                self.ke[i, j] = kz[ie, je]

        # Set the matrices for recovery of kappa_y = d^2 v / dx^2 and
        #  kappa_z = d^2 w / dx^2
        self.By = np.zeros(8)
        self.Bz = np.zeros(8)
        by = np.array([0.0, -1.0 / Le, 0.0, 1.0 / Le])
        for ie, i in enumerate(xydof):
            self.By[i] = by[ie]

        bz = np.array([0.0, 1.0 / Le, 0.0, -1.0 / Le])
        for ie, i in enumerate(xzdof):
            self.Bz[i] = bz[ie]

        # Compute the mass matrix
        m0 = (
            np.array(
                [
                    [156.0, 22.0, 54.0, -13.0],
                    [22.0, 4.0, 13.0, -3.0],
                    [54.0, 13.0, 156.0, -22.0],
                    [-13.0, -3.0, -22.0, 4.0],
                ]
            )
            / 420.0
        )
        my = (m0 * Le) * np.outer(cxy, cxy)
        mz = (m0 * Le) * np.outer(cxz, cxz)

        # Set the elements into the stiffness matrix
        self.me = np.zeros((8, 8))
        for ie, i in enumerate(xydof):
            for je, j in enumerate(xydof):
                self.me[i, j] = my[ie, je]

        for ie, i in enumerate(xzdof):
            for je, j in enumerate(xzdof):
                self.me[i, j] = mz[ie, je]

        return

    def eval_bernstein(self, u, order):
        """
        Evaluate the Bernstein polynomial basis functions at the given parametric locations
        Parameters
        ----------
        u : np.ndarray
            Parametric locations for the basis functions
        order : int
            Order of the polynomial
        Returns
        -------
        N : np.ndarray
            Matrix mapping the design inputs to outputs
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

    def get_vars(self, elem, u=None):
        """
        Get the global variables for the given element
        Parameters
        ----------
        elem : int
            Element index
        u : np.ndarray
            Global variables
        Returns
        -------
        elem_vars : list
            List of length 8 of the associated element variables
        """
        if elem == 0:
            elem_vars = [-1, -1, -1, -1, 0, 1, 2, 3]
        elif elem == self.nelems - 1:
            i = 4 * (elem - 1)
            elem_vars = [i, i + 1, i + 2, i + 3, -1, -1, -1, -1]
        else:
            i = 4 * (elem - 1)
            j = 4 * elem
            elem_vars = [i, i + 1, i + 2, i + 3, j, j + 1, j + 2, j + 3]

        if u is None:
            return elem_vars
        else:
            elem_u = np.zeros(8)
            for ie, i in enumerate(elem_vars):
                if i >= 0:
                    elem_u[ie] = u[i]
            return elem_u

    def get_sectional_mass(self, x):
        """
        Given the design variables, compute the sectional mass -
        the mass per unit length of the beam
        Parameters
        ----------
        x : np.ndarray
            The design variables
        Returns
        -------
        rhoA : np.ndarray
            The piecewise constant mass per unit length of the beam in each element
        """

        # Compute the inner radius
        inner_radius = np.dot(self.N, x)

        # Compute the sectional mass
        t0 = self.t + inner_radius
        rhoA = self.density * np.pi * (t0**2 - inner_radius**2)

        return rhoA

    def get_sectional_mass_deriv(self, dfdrhoA):
        """
        Given the derivative of a function w.r.t. rhoA, compute dfdx
        Parameters
        ----------
        x : np.ndarray
            The design variables
        dfdrhoA : np.ndarray
            The derivative of a function w.r.t. rhoA
        Returns
        -------
        dfdx : np.ndarray
            The derivative of the function w.r.t. x

            dfdr = 2 * density * pi * t * dfdrhoA
            drdx = N
            dfdx = dfdr * drdx

        """

        dfdr = 2.0 * self.density * np.pi * self.t * dfdrhoA
        drdx = self.N
        dfdx = np.dot(dfdr, drdx)

        return dfdx

    def get_mass(self, x):
        """
        Get the mass of the beam

            secional_mass = density * pi * (r_outer**2 - r_inner**2)

            mass = Le * sum(secional_mass)
        """

        Le = self.L / self.nelems

        return Le * np.sum(self.get_sectional_mass(x))

    def get_mass_deriv(self):
        """
        Get the derivative of the mass of the beam
        """

        Le = self.L / self.nelems
        dfdrhoA = Le * np.ones(self.nelems)

        return self.get_sectional_mass_deriv(dfdrhoA)

    def get_mass_matrix(self, x):
        """
        Compute the mass matrix for the clamped-clamped beam
        """

        rhoA = self.get_sectional_mass(x)

        M = np.zeros((self.ndof, self.ndof), dtype=rhoA.dtype)
        for k in range(self.nelems):
            elem_vars = self.get_vars(k)
            for ie, i in enumerate(elem_vars):
                for je, j in enumerate(elem_vars):
                    if i >= 0 and j >= 0:
                        M[i, j] += rhoA[k] * self.me[ie, je]

        return M

    def get_mass_matrix_deriv(self, u, v):
        dfdrhoA = np.zeros((self.nelems))

        for k in range(self.nelems):
            elem_vars = self.get_vars(k)
            for ie, i in enumerate(elem_vars):
                for je, j in enumerate(elem_vars):
                    if i >= 0 and j >= 0:
                        dfdrhoA[k] += u[i] * v[j] * self.me[ie, je]

        return self.get_sectional_mass_deriv(dfdrhoA)

    def get_sectional_stiffness(self, x):
        """
        Given the design variables, compute the sectional stiffness
        """

        # Compute the inner radius
        inner_radius = np.dot(self.N, x)

        # Compute the sectional bending stiffness
        t0 = self.t + inner_radius
        EI = self.E * np.pi * (t0**4 - inner_radius**4)

        return EI

    def get_sectional_stiffness_deriv(self, x, dfdEI):
        """
        Given the design variables, compute the sectional stiffness
        """

        # Compute the inner radius
        inner_radius = np.dot(self.N, x)

        # Compute the sectional bending stiffness
        t0 = self.t + inner_radius

        dfdx = (
            4.0
            * self.E
            * np.pi
            * np.dot(self.N.T, (t0**3 - inner_radius**3) * dfdEI)
        )

        return dfdx

    def get_stiffness_matrix(self, x):
        """
        Compute the stiffness matrix of the clamped-clamped beam
        """

        EI = self.get_sectional_stiffness(x)

        K = np.zeros((self.ndof, self.ndof), dtype=EI.dtype)
        for k in range(self.nelems):
            elem_vars = self.get_vars(k)
            for ie, i in enumerate(elem_vars):
                for je, j in enumerate(elem_vars):
                    if i >= 0 and j >= 0:
                        K[i, j] += EI[k] * self.ke[ie, je]

        return K

    def get_stiffness_matrix_deriv(self, x, u, v):
        dfdEI = np.zeros((self.nelems))
        for k in range(self.nelems):
            elem_vars = self.get_vars(k)
            for ie, i in enumerate(elem_vars):
                for je, j in enumerate(elem_vars):
                    if i >= 0 and j >= 0:
                        dfdEI[k] += u[i] * v[j] * self.ke[ie, je]

        return self.get_sectional_stiffness_deriv(x, dfdEI)

    def solve_full_eigenvalue_problem(self, x):
        A = self.get_stiffness_matrix(x)
        B = self.get_mass_matrix(x)

        # Compute the eigenvalues of the generalized eigen problem
        lam, Q = eigh(A, B)

        return lam, Q

    def solve_eigenvalue_problem(self, x):
        A = self.get_stiffness_matrix(x)
        B = self.get_mass_matrix(x)

        # Compute the eigenvalues of the generalized eigen problem
        lam, Q = eigh(A, B)

        return lam[: self.Np], Q[:, : self.Np]

    def appox_min_eigenvalue(self, x):
        lam, Q = self.solve_eigenvalue_problem(x)

        min_lam = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - min_lam))
        return min_lam - np.log(np.sum(eta)) / self.ksrho

    def approx_min_eigenvalue_deriv(self, x):
        lam, Q = self.solve_eigenvalue_problem(x)

        min_lam = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - min_lam))
        eta = eta / np.sum(eta)
        dfdx = np.zeros(self.ndvs)

        for k in range(self.Np):
            dfdx += eta[k] * (
                self.get_stiffness_matrix_deriv(x, Q[:, k], Q[:, k])
                - lam[k] * self.get_mass_matrix_deriv(Q[:, k], Q[:, k])
            )

        return dfdx

    def exact_eigenvector2(self, x):
        """
        Compute the eigenvector constraint
        h = tr(D * B^{-1} * exp(- rho * A * B^{-1})/ tr(exp(- rho * A * B^{-1}))
        """

        A = self.get_stiffness_matrix(x)
        B = self.get_mass_matrix(x)

        Binv = np.linalg.inv(B)
        exp = expm(-self.ksrho * np.dot(A, Binv))
        h = np.trace(np.dot(self.D, np.dot(Binv, exp))) / np.trace(exp)

        return h

    def exact_eigenvector(self, x):
        """
        Compute the eigenvector constraint

        h = tr(eta * Q^T * D * Q)
        """

        A = self.get_stiffness_matrix(x)
        B = self.get_mass_matrix(x)

        # Compute the eigenvalues of the generalized eigen problem
        lam, Q = eigh(A, B)

        eta = np.exp(-self.ksrho * (lam - np.min(lam)))
        eta = eta / np.sum(eta)

        # h = np.trace(np.dot(np.diag(eta), np.dot(Q.T, np.dot(self.D, Q))))
        h = np.trace(np.diag(eta) @ Q.T @ self.D @ Q)
        return h

    def approx_eigenvector(self, x):
        lam, QN = self.solve_eigenvalue_problem(x)
        eta = np.exp(-self.ksrho * (lam - np.min(lam)))
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(self.Np):
            h += eta[i] * np.dot(QN[:, i], np.dot(self.D, QN[:, i]))

        return h

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

    def exact_eigenvector_deriv(self, x):
        """
        Compute the exact derivative
        """

        lam, Q = self.solve_full_eigenvalue_problem(x)

        lam_min = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - lam_min))
        trace = np.sum(eta)
        eta = eta / trace

        h = 0.0
        for i in range(Q.shape[0]):
            h += eta[i] * np.dot(Q[:, i], np.dot(self.D, Q[:, i]))

        dfdx = np.zeros(self.ndvs)

        G = np.dot(np.diag(eta), np.dot(Q.T, np.dot(self.D, Q)))
        for j in range(Q.shape[0]):
            for i in range(Q.shape[0]):
                qDq = np.dot(Q[:, i], np.dot(self.D, Q[:, j]))
                scalar = qDq
                if i == j:
                    scalar = qDq - h

                Eij = scalar * self.precise(self.ksrho, trace, lam_min, lam[i], lam[j])

                # Add to dfdx from A
                dfdx += Eij * self.get_stiffness_matrix_deriv(x, Q[:, i], Q[:, j])

                # Add to dfdx from B
                dfdx -= (Eij * lam[j] + G[i, j]) * self.get_mass_matrix_deriv(
                    Q[:, i], Q[:, j]
                )

        return dfdx

    def approx_eigenvector_deriv(self, x):
        """
        Approximately compute the forward derivative
        """

        lam, QN = self.solve_eigenvalue_problem(x)

        lam_min = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - lam_min))
        trace = np.sum(eta)
        eta = eta / trace

        # Compute the h values
        h = 0.0
        for i in range(self.Np):
            h += eta[i] * np.dot(QN[:, i], np.dot(self.D, QN[:, i]))

        # Set the value of the derivative
        dfdx = np.zeros(self.ndvs)

        for j in range(self.Np):
            for i in range(self.Np):
                qDq = np.dot(QN[:, i], np.dot(self.D, QN[:, j]))
                scalar = qDq
                if i == j:
                    scalar = qDq - h

                Eij = scalar * self.precise(self.ksrho, trace, lam_min, lam[i], lam[j])

                # Add to dfdx from A
                dfdx += Eij * self.get_stiffness_matrix_deriv(x, QN[:, i], QN[:, j])

                # Add to dfdx from B
                dfdx -= Eij * lam[i] * self.get_mass_matrix_deriv(QN[:, i], QN[:, j])

        # Get the stiffness and mass matrices
        A = self.get_stiffness_matrix(x)
        B = self.get_mass_matrix(x)

        # Form the augmented linear system of equations
        for k in range(self.Np):
            # Compute B * vk = D * qk
            bk = np.dot(self.D, QN[:, k])
            vk = np.linalg.solve(B, -eta[k] * bk)
            dfdx += self.get_mass_matrix_deriv(QN[:, k], vk)

            # Solve the augmented system of equations for wk
            Ak = A - lam[k] * B
            Ck = np.dot(B, QN)

            # Set up the augmented linear system of equations
            mat = np.block([[Ak, Ck], [Ck.T, np.zeros((self.Np, self.Np))]])
            b = np.zeros(mat.shape[0])

            # Compute the right-hand-side vector
            b[: self.ndof] = -eta[k] * bk

            # Solve the first block linear system of equations
            sol = np.linalg.solve(mat, b)
            wk = sol[: self.ndof]

            # Compute the contributions from the derivative from Adot
            dfdx += 2.0 * self.get_stiffness_matrix_deriv(x, QN[:, k], wk)

            # Add the contributions to the derivative from Bdot here...
            dfdx -= lam[k] * self.get_mass_matrix_deriv(QN[:, k], wk)

            # Now, compute the remaining contributions to the derivative from
            # B by solving the second auxiliary system
            dk = np.dot(A, vk)
            b[: self.ndof] = dk

            sol = np.linalg.solve(mat, b)
            uk = sol[: self.ndof]

            # Compute the contributions from the derivative
            dfdx -= self.get_mass_matrix_deriv(QN[:, k], uk)

        return dfdx

    def get_stress_values(self, x, eta, Q, allowable=1.0):
        # Compute the inner radius
        inner_radius = np.dot(self.N, x)

        # Compute the sectional bending stiffness
        t0 = self.t + inner_radius

        # Loop over all the eigenvalues
        stress = np.zeros(self.nelems)

        for elem in range(self.nelems):
            # Get the element variables
            elem_vars = self.get_vars(elem)

            for k in range(self.Np):
                ky = 0.0
                kz = 0.0
                for ie, i in enumerate(elem_vars):
                    if i >= 0:
                        # Compute ky = d^2v/dx^2
                        ky += self.By[ie] * Q[i, k]

                        # Compute kz = d^2w/dx^2
                        kz += self.Bz[ie] * Q[i, k]

                # Comute the von-Mises stress squared and sum contributions from
                # the top and
                r0 = t0[elem]  # The outer-radius

                # Compute the stress at two points
                sx1 = self.E * ky * r0
                sx2 = self.E * kz * r0

                # Sum the von Mises stress squared
                von_mises2 = sx1**2 + sx2**2

                # Add the values - this is eta[k] * qk^{T} * Di * qk
                stress[elem] += eta[k] * (von_mises2 / allowable**2)

        return stress

    def get_stress_values_deriv(self, x, eta_stress, eta, Q, allowable=1.0):
        # Compute the inner radius
        inner_radius = np.dot(self.N, x)

        # Compute the sectional bending stiffness
        t0 = self.t + inner_radius

        dfdt0 = np.zeros(self.nelems)

        for elem in range(self.nelems):
            # Get the element variables
            elem_vars = self.get_vars(elem)

            for k in range(self.Np):
                ky = 0.0
                kz = 0.0
                for ie, i in enumerate(elem_vars):
                    if i >= 0:
                        # Compute ky = d^2v/dx^2
                        ky += self.By[ie] * Q[i, k]

                        # Compute kz = d^2w/dx^2
                        kz += self.Bz[ie] * Q[i, k]

                # Comute the von-Mises stress squared and sum contributions from
                # the top and
                r0 = t0[elem]  # The outer-radius

                # Compute the stress at two points
                sx1 = self.E * ky * r0
                sx2 = self.E * kz * r0

                # Sum the von Mises stress squared
                dvon_mises2 = 2.0 * self.E * (sx1 * ky + sx2 * kz)

                # Add the values - this is eta[k] * qk^{T} * Di * qk
                dfdt0[elem] += (
                    eta_stress[elem] * eta[k] * (dvon_mises2 / allowable**2)
                )

        return np.dot(self.N.T, dfdt0)

    def get_stress_product(self, x, eta_stress, Qk, allowable=1.0):
        # Compute the inner radius
        inner_radius = np.dot(self.N, x)

        # Compute the sectional bending stiffness
        t0 = self.t + inner_radius

        product = np.zeros(self.ndof)

        for elem in range(self.nelems):
            # Get the element variables
            elem_vars = self.get_vars(elem)

            ky = 0.0
            kz = 0.0
            for ie, i in enumerate(elem_vars):
                if i >= 0:
                    # Compute ky = d^2v/dx^2
                    ky += self.By[ie] * Qk[i]

                    # Compute kz = d^2w/dx^2
                    kz += self.Bz[ie] * Qk[i]

            # Comute the von-Mises stress squared and sum contributions from
            # the top and
            r0 = t0[elem]  # The outer-radius

            # Compute the stress at two points
            sx1 = self.E * ky * r0
            sx2 = self.E * kz * r0

            for ie, i in enumerate(elem_vars):
                if i >= 0:
                    product[i] += (
                        eta_stress[elem]
                        * (self.E * r0)
                        * (sx1 * self.By[ie] + sx2 * self.Bz[ie])
                    ) / allowable**2

        return product

    def eigenvector_stress(self, x, rho=10.0, allowable=1.0):
        """
        Compute the aggregated stress value based on the lowest mode shapes
        """

        # Solve the eigenvalue problem
        lam, QN = self.solve_eigenvalue_problem(x)

        lam_min = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - lam_min))
        trace = np.sum(eta)
        eta = eta / trace

        # Compute the stresses in the beam
        stress = self.get_stress_values(x, eta, QN, allowable=allowable)

        # Now aggregate over the stress
        h = np.max(stress) + (1 / rho) * np.log(
            np.sum(np.exp(rho * (stress - np.max(stress))))
        )

        return h

    def exact_eigenvector_stress_deriv(self, x, rho=10.0, allowable=1.0):
        # Solve the eigenvalue problem
        lam, Q = self.solve_full_eigenvalue_problem(x)

        # Get the stiffness and mass matrices
        A = self.get_stiffness_matrix(x)
        B = self.get_mass_matrix(x)
        U, _ = np.linalg.qr(B @ Q)
        Z = np.eye(np.shape(A)[0]) - U @ U.T

        # Compute the eta values
        lam_min = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - lam_min))
        trace = np.sum(eta)
        eta = eta / trace

        # Compute the stresses in the beam
        stress = self.get_stress_values(x, eta, Q, allowable=allowable)

        # Now aggregate over the stress
        max_stress = np.max(stress)
        eta_stress = np.exp(rho * (stress - max_stress))
        eta_stress = eta_stress / np.sum(eta_stress)

        h = np.dot(eta_stress, stress)
        hdot = self.get_stress_values_deriv(x, eta_stress, eta, Q, allowable=allowable)

        for j in range(np.shape(Q)[0]):
            # Compute D * Q[:, j]
            Dqj = self.get_stress_product(x, eta_stress, Q[:, j], allowable=allowable)

            for i in range(np.shape(Q)[0]):
                Adot_q = self.get_stiffness_matrix_deriv(x, Q[:, i], Q[:, j])
                Bdot_q = self.get_mass_matrix_deriv(Q[:, i], Q[:, j])
                Dq = Q[:, i] @ Dqj
                Eij = self.precise(self.ksrho, trace, lam_min, lam[i], lam[j])

                if i == j:
                    hdot += Eij * (Dq - h) * (Adot_q - lam[j] * Bdot_q)
                else:
                    hdot += Eij * Dq * (Adot_q - lam[j] * Bdot_q)

                hdot -= eta[i] * Dq * Bdot_q

        return hdot

    def eigenvector_stress_deriv(self, x, rho=10.0, allowable=1.0):
        # Solve the eigenvalue problem
        lam, Q = self.solve_full_eigenvalue_problem(x)

        # Compute the eta values
        lam_min = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - lam_min))
        trace = np.sum(eta)
        eta = eta / trace

        # count nonzero entries in the eta vector as self.Np
        # self.Np = np.count_nonzero(eta)

        # resize Q to self.Np columns
        QN = Q[:, : self.Np]

        # Get the stiffness and mass matrices
        A = self.get_stiffness_matrix(x)
        B = self.get_mass_matrix(x)
        U, _ = np.linalg.qr(B @ QN)
        Z = np.eye(np.shape(A)[0]) - U @ U.T

        # Compute the stresses in the beam
        stress = self.get_stress_values(x, eta, QN, allowable=allowable)

        # Now aggregate over the stress
        max_stress = np.max(stress)
        eta_stress = np.exp(rho * (stress - max_stress))
        eta_stress = eta_stress / np.sum(eta_stress)

        h = np.dot(eta_stress, stress)
        hdot = self.get_stress_values_deriv(x, eta_stress, eta, QN, allowable=allowable)

        for j in range(self.Np):
            # Compute D * QN[:, j]
            Dqj = self.get_stress_product(x, eta_stress, QN[:, j], allowable=allowable)

            for i in range(j + 1):
                Adot_q = self.get_stiffness_matrix_deriv(x, QN[:, i], QN[:, j])
                Bdot_q = self.get_mass_matrix_deriv(QN[:, i], QN[:, j])
                Dq = QN[:, i] @ Dqj
                Eij = self.precise(self.ksrho, trace, lam_min, lam[i], lam[j])

                if i == j:
                    hdot += Eij * (Dq - h) * (Adot_q - lam[j] * Bdot_q)
                else:
                    hdot += Eij * Dq * (2 * Adot_q - (lam[i] + lam[j]) * Bdot_q)

        # Form the augmented linear system of equations
        for j in range(self.Np):
            # solve the first linear system
            rhs1 = -eta[j] * self.get_stress_product(
                x, eta_stress, QN[:, j], allowable=allowable
            )
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

            # ic(j, lam[j])

            # ic(j, Ak)
            # ic(j, -self.ksrho * (lam[j] - lam_min))
            # ic(j, eta[j])
            # ic(j, uj)
            # ic(j, vj)
            # ic(j, wj)

            # Compute the contributions from the derivative from Adot
            hdot += 2.0 * self.get_stiffness_matrix_deriv(x, QN[:, j], vj)
            hdot += self.get_mass_matrix_deriv(QN[:, j], uj)
            hdot -= self.get_mass_matrix_deriv(QN[:, j], lam[j] * vj)
            hdot -= self.get_mass_matrix_deriv(QN[:, j], wj)

        return hdot

    def check_consisten(self, x, QN, eta, eta_stress, allowable=1.0):
        # Compute the stresses in the beam
        stress = self.get_stress_values(x, eta, QN, allowable=allowable)

        value = np.dot(stress, eta_stress)
        print("value = ", value)

        value2 = 0.0
        for k in range(self.Np):
            prod = self.get_stress_product(x, eta_stress, QN[:, k], allowable=allowable)
            value2 += eta[k] * np.dot(prod, QN[:, k])

        print("value2 = ", value2)
        print("rel. diff = ", (value - value2) / value)

        return

    # def plot_modes(self, x, N=5):
    #     """
    #     Plot the modes
    #     """

    #     u = np.linspace(0.5 / self.nelems, 1.0 - 0.5 / self.nelems, self.nelems)
    #     xvals = np.dot(self.N, x)

    #     plt.figure()
    #     plt.plot(u, xvals)

    #     _, QN = self.solve_eigenvalue_problem(x, N=N)

    #     plt.figure()
    #     for k in range(N):
    #         u = np.zeros(self.nelems + 1)
    #         x = np.linspace(0, self.L, self.nelems + 1)
    #         u[1:-1] = QN[::4, k]
    #         if k == 1:
    #             plt.plot(x, u)

    #     plt.show()

    def process_Q(self, Q, scale=4.0):
        """
        Process:
            Q  ->  (dy, dz, ay, az)

        Parameters
            Q: the modes

            dy: displacement in y
            dz: displacement in z
            ay: rotation along y-axis
            az: rotation along z-axis
        """

        # scale the modes to be more visible
        Q *= scale * self.L / np.max(Q)

        dy, dz, ay, az = Q[0::4, :], Q[1::4, :], Q[2::4, :], Q[3::4, :]
        ay /= 180.0 / np.pi  # convert to radians
        az /= 180.0 / np.pi  # convert to radians

        # add boundary conditions to dy, dz, ay, az
        tmp = [dy, dz, ay, az]
        for i in range(len(tmp)):
            tmp[i] = np.concatenate((np.zeros((1, tmp[i].shape[1])), tmp[i]))
            tmp[i] = np.concatenate((tmp[i], np.zeros((1, tmp[i].shape[1]))))

        dy, dz, ay, az = tmp

        return dy, dz, ay, az

    def process_r(self, r):
        """
        Process:
            r_element  ->  r_node

            r_node[i+1] = (r[i] + r[i+1]) / 2
        """

        # caculate the values of r from the midpoints of the elements to the nodes
        r_node = np.zeros(self.nelems + 1)
        for i in range(self.nelems - 1):
            r_node[i + 1] = (r[i + 1] + r[i]) / 2
        r_node[0] = 2 * r[0] - r_node[1]
        r_node[-1] = 2 * r[-1] - r_node[-2]

        return r_node

    def converter(self, dy, dz, ay, az, r, degree_start=0, degree_end=2 * np.pi):
        """
        Convert the coordinates of the tube

            from (dy, dz, ay, az) -> (x, y, z)
        """

        theta = np.linspace(degree_start, degree_end, np.power(2, 6))
        x = np.linspace(0, self.L, np.shape(r)[0])
        x, theta = np.meshgrid(x, theta)

        # displacement
        y = dy + r * np.cos(theta)
        z = dz + r * np.sin(theta)

        # rotation
        x = x - y * np.sin(az) + z * np.cos(az) * np.sin(ay)
        y = y * np.cos(az) + z * np.sin(az) * np.sin(ay)
        z = z * np.cos(ay)

        return x, y, z

    def plot_tube(self, ax, x):
        """
        Plot the tube with inner surface and outer surface

            r: the inner radius of the tube
            t: the thickness of the tube
            L: the length of the tube
        """

        # process the data r_element -> r_node
        r = np.dot(self.N, x)
        r = self.process_r(r)

        # ignore the displacement and rotation
        n_nodes = r.shape[0]
        dy = np.zeros(n_nodes)
        dz = np.zeros(n_nodes)
        ay = np.zeros(n_nodes)
        az = np.zeros(n_nodes)

        # convert the mode to (x, y, z)
        deg0 = -np.pi
        deg1 = 0.5 * np.pi

        x_in, y_in, z_in = self.converter(dy, dz, ay, az, r, deg0, deg1)
        x_out, y_out, z_out = self.converter(dy, dz, ay, az, r + self.t, deg0, deg1)

        # # plot the surface
        ax.plot_surface(x_in, y_in, z_in, color="b", alpha=0.8)
        ax.plot_surface(x_out, y_out, z_out, color="b", alpha=0.2)
        ax.set_axis_off()

        return ax

    def plot_modes(self, ax, x, n=2, scale=4.0):
        """
        Plot n modes of the tube

            Q: the modes
            L: the length of the tube
            n: the number of modes to plot
        """

        _, Q = self.solve_eigenvalue_problem(x)

        # check if n is less than the number of columns
        if n > Q.shape[1]:
            print("Warning: n is too large, set n to the number of columns")
            n = Q.shape[1]
        if n > 8:
            print("Warning: n is too large, set n to 8")
            n = 8

        # set r to constant
        r = 0.0025 * np.ones(Q.shape[0] // 4 + 2) * self.L

        # process the data from (Q, r) -> (dy, dz, ay, az, r_node)
        dy, dz, ay, az = self.process_Q(Q[:, :n], scale=scale)

        color = ["b", "b", "r", "r", "g", "g", "y", "y"]
        # plot the tube
        for i in range(n):
            # convert the coordinates of the tube
            x, y, z = self.converter(dy[:, i], dz[:, i], ay[:, i], az[:, i], r)
            # each two surfeces have the same color
            ax.plot_surface(x, y, z, color=color[i], alpha=0.8)

            # add dashed lines to show the displacement
            n_pecks = (np.floor(i / 2) + 1).astype(int)
            max_dy = np.sort(np.abs(dy[:, i]))[-n_pecks:]
            max_dz = np.sort(np.abs(dz[:, i]))[-n_pecks:]
            max_index = np.zeros(n_pecks).astype(int)
            for j in range(n_pecks):
                if np.max(max_dy) > np.max(max_dz):
                    max_index[j] = np.where(np.abs(dy[:, i]) == np.max(max_dy[j]))[0][0]

                else:
                    max_index[j] = np.where(np.abs(dz[:, i]) == np.max(max_dz[j]))[0][0]

                y_dash = np.linspace(0, dy[max_index[j], i], 100)
                z_dash = np.linspace(0, dz[max_index[j], i], 100)
                x_dash = np.linspace(x[0, max_index[j]], x[0, max_index[j]], 100)
                ax.plot(
                    x_dash,
                    y_dash,
                    z_dash,
                    color="k",
                    linestyle="--",
                    linewidth=0.5,
                    alpha=0.5,
                )
                if i < 2:
                    ax.scatter(
                        x[0, max_index[j]],
                        dy[max_index[j], i],
                        dz[max_index[j], i],
                        color=color[i],
                        s=2,
                    )
                    ax.scatter(x[0, max_index[j]], 0, 0, color=color[i], s=2)
                else:
                    ax.scatter(
                        x[0, max_index[j]],
                        dy[max_index[j], i],
                        dz[max_index[j], i],
                        color=color[i],
                        s=2,
                        label="upper curvature",
                    )
                    ax.scatter(x[0, max_index[j]], 0, 0, color=color[i], s=2)

        ax.legend(
            [Line2D([0], [0], color=color[i], lw=1) for i in range(n)],
            ["mode {}".format(i + 1) for i in range(n)],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.08),
            # decrease the size of the legend box
            # loc="best",
            # bbox_to_anchor=(1, 0.85),
            ncol=n,
            borderaxespad=0,
            frameon=False,
            # size of box
            # prop={"size": 5},
            borderpad=0.5,
        )

        return ax

    def plot_stress(self, ax, res_x):
        """
        Plot the tube with inner surface and outer surface

            r: the inner radius of the tube
            t: the thickness of the tube
            L: the length of the tube
        """

        # process the data r_element -> r_node
        r = np.dot(self.N, res_x)
        r = self.process_r(r)

        # get the stress
        lam, QN = self.solve_eigenvalue_problem(res_x)

        # compute eta
        eta = np.exp(-self.ksrho * (lam - np.min(lam)))
        eta = eta / np.sum(eta)

        stress = self.get_stress_values(res_x, eta, QN)
        stress = self.process_r(stress)

        # normalize the stress to 0-1
        stress = (stress - np.min(stress)) / (np.max(stress) - np.min(stress))

        theta = np.linspace(-np.pi, 0.5 * np.pi, np.power(2, 6))
        x = np.linspace(0, self.L, np.shape(r)[0])

        # interpolate the surface
        kind = "cubic"  # kind = "linear"
        r = interpolate.interp1d(x, r, kind=kind)
        stress = interpolate.interp1d(x, stress, kind=kind)

        xnew = np.linspace(0, self.L, np.shape(x)[0] * 4)
        rnew = r(xnew)
        stress = stress(xnew)
        x, theta = np.meshgrid(xnew, theta)

        # compute the coordinates of the tube
        y_in = rnew * np.cos(theta)
        z_in = rnew * np.sin(theta)

        y_out = (rnew + self.t) * np.cos(theta)
        z_out = (rnew + self.t) * np.sin(theta)

        # reshape the stress with repeated x_in.shape[0] times column
        stress_surf = np.tile(stress, (x.shape[0], 1))

        # normalize the stress
        # norm = mpl.colors.Normalize(vmin=np.min(stress), vmax=np.max(stress))
        cmap = mpl.cm.coolwarm  # cmap = mpl.cm.Greys, cmap = mpl.cm.Purples
        m = cm.ScalarMappable(cmap=cmap)

        ax.plot_surface(
            x,
            y_in,
            z_in,
            rstride=1,
            cstride=1,
            facecolors=cmap(stress_surf),
            linewidth=0.0,
            alpha=0.5,
        )

        ax.plot_surface(
            x,
            y_out,
            z_out,
            rstride=1,
            cstride=1,
            facecolors=cmap(stress_surf),
            linewidth=0.0,
            alpha=0.2,
        )

        # plot the colorbar
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.05,
                ax.get_position().y0 + 0.15,
                0.01,
                ax.get_position().height - 0.4,
            ]
        )
        cb = plt.colorbar(m, shrink=1, aspect=1, cax=cax)
        cb.set_label("Normalized \n ks-stress", labelpad=-12, y=1.15, rotation=0)

        # add points to the plot
        # lowest_stress = np.sort(np.abs(stress))[0:2]
        # lowest_stress_index = np.zeros(2).astype(int)
        # for i in range(len(lowest_stress)):
        #     lowest_stress_index[i] = np.where(np.abs(stress) == lowest_stress[i])[0][0]
        # ax.scatter(xnew[lowest_stress_index], 0, 0, color="b", s=2)
        ax.scatter(xnew[-1], 0, 0, color="r", s=4)
        ax.scatter(0, 0, 0, color="r", s=4)
        # add legend
        plt.legend(
            [
                Line2D([0], [0], color="b", marker="o", lw=0, markersize=2),
                Line2D([0], [0], color="r", marker="o", lw=0, markersize=2),
            ],
            ["lower curvature", "higher curvature"],
            # loc="lower center",
            bbox_to_anchor=(-25, -0.15),
            # decrease the size of the legend box
            # loc="best",
            # bbox_to_anchor=(1, 0.85),
            ncol=2,
            borderaxespad=0,
            frameon=False,
            # size of box
            # prop={"size": 5},
            borderpad=0.5,
        )

        # ax.set_axis_off()
        # don't show the value of the axis
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # don't show the axis line
        ax.xaxis.line.set_lw(0.0)
        ax.yaxis.line.set_lw(0.0)
        ax.zaxis.line.set_lw(0.0)

        return ax


np.random.seed(12345)


# problem = "exact_derivative"
# problem = "optimization_eigenvalue"
# problem = "optimization_eigenvector"
problem = "stress"

if problem == "exact_derivative":
    # sets for the beam
    set_beam = {
        "nelems": 11,
        "ndvs": 5,
        "L": 1.0,
        "t": 0.025,
        "N": 5,
        "ksrho": 10.0,
    }

    npts = 10
    nlines = 12
    res = np.zeros((npts, nlines))

    # Pick a direction
    x = 0.1 * np.ones(set_beam["ndvs"])
    px = np.ones(set_beam["ndvs"])

    beam0 = EulerBeam(**set_beam)
    lam, Q = beam0.solve_eigenvalue_problem(x)
    ksrho = lam[0] * (10 ** np.linspace(-5, 1, npts))

    for i in range(npts):
        for j in range(2, nlines):
            beam = EulerBeam(
                set_beam["nelems"],
                set_beam["ndvs"],
                set_beam["L"],
                set_beam["t"],
                j - 1,
                ksrho[i],
            )

            # Set the matrix component we want to zero
            dof = np.arange(0, beam.nelems // 4)
            beam.D[dof, dof] = 1.0

            dh = 1e-30

            if j == 2:
                res[i, 0] = np.dot(px, beam.exact_eigenvector_deriv(x))
                res[i, 1] = beam.exact_eigenvector(x + 1j * dh * px).imag / dh

            res[i, j] = np.dot(px, beam.approx_eigenvector_deriv(x))
        print(".", end="", flush=True)
    print("")

    # save the data
    np.savez(
        "output/exact_derivative.npz",
        ksrho=ksrho,
        exact=res[:, 0],
        fd=res[:, 1],
        approx1=res[:, 2],
        approx2=res[:, 3],
        approx3=res[:, 4],
        approx4=res[:, 5],
        approx5=res[:, 6],
        approx6=res[:, 7],
        approx7=res[:, 8],
        approx8=res[:, 9],
        approx9=res[:, 10],
        approx10=res[:, 11],
    )

    # read the data
    data = np.load("output/exact_derivative.npz")
    ksrho = data["ksrho"]
    fd = data["fd"]
    exact = data["exact"]

    with plt.style.context(["science", "nature"]):
        # plt.style.use(niceplots.get_style())
        # plt.style.use("mystyle")
        # plt.figure(figsize=(8, 6))
        # plt.semilogx(ksrho, fd, label="complex step")
        # plt.semilogx(ksrho, exact, label="exact")
        # for i in range(1, nlines - 1):
        #     plt.semilogx(ksrho, data["approx%d" % i], label="approx N = %d" % i)
        # plt.legend()
        # plt.savefig("output/exact_derivative.png")

        colors = ["k", "b", "r", "b", "r", "b", "r", "b", "r", "b"]
        styles = ["-", "-", "--", "-", "--", "-", "--", "-", "--", "-"]
        alpha = [1.0, 1.0, 1.0, 0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2]
        fig, ax = plt.subplots()
        for i in range(1, nlines - 1):
            plt.loglog(
                ksrho,
                np.abs(data["approx%d" % i] - exact) / np.abs(exact),
                label="%d" % i,
                color=colors[i - 1],
                alpha=alpha[i - 1],
                linestyle=styles[i - 1],
            )
        # niceplots.adjust_spines(ax)
        handles, labels = ax.get_legend_handles_labels()
        handles = [handles[i] for i in range(0, len(handles), 2)] + [
            handles[i] for i in range(1, len(handles), 2)
        ]
        labels = [labels[i] for i in range(0, len(labels), 2)] + [
            labels[i] for i in range(1, len(labels), 2)
        ]
        ax.legend(
            handles,
            labels,
            title="Approximate: N",
            ncol=2,
            loc="upper right",
        )
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel("Relative Error")
        ax.tick_params(direction="out")
        ax.tick_params(which="minor", direction="out")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        plt.savefig("output/Fig1.pdf")

        # ax.legend(title="Approximate: N", ncol=2, loc="upper right")
        # ax.set_xlabel(r"$\rho$")
        # ax.set_ylabel("Relative Error")
        # ax.tick_params(direction="out")
        # ax.tick_params(which="minor", direction="out")
        # plt.savefig("output/Fig1.pdf")


elif problem == "optimization_eigenvector":
    # sets for the beam
    set_beam = {
        # set nelems even to make middle node is the one we want to optimize
        "nelems": 50,
        "ndvs": 50,
        "L": 1.0,
        "t": 0.025,
        "N": 10,
        "ksrho": 10.0,
    }

    beam = EulerBeam(**set_beam)

    node_index = np.floor(beam.nelems / 2).astype(int)
    beam.D[(node_index - 1) * 4, (node_index - 1) * 4] = 1.0
    beam.D[(node_index - 1) * 4 + 1, (node_index - 1) * 4 + 1] = 1.0

    # sets for the optimization
    set_opt = {
        "x0": 0.05 * beam.L * np.ones(beam.ndvs),
        "x_con": 0.075 * beam.L * np.ones(beam.ndvs),
        "x_lower": 0.01 * beam.L * np.ones(beam.ndvs),
        "x_upper": 0.1 * beam.L * np.ones(beam.ndvs),
    }

    # minimize the eigenvector
    obj = lambda x: 0.1 * beam.approx_eigenvector(x)
    obj_grad = lambda x: 0.1 * beam.approx_eigenvector_deriv(x)

    # constrain the mass
    mass = lambda x: beam.get_mass(set_opt["x_con"]) - beam.get_mass(x)
    mass_grad = lambda x: beam.get_mass_deriv()

    res = minimize(
        obj,
        set_opt["x0"],
        jac=obj_grad,
        method="SLSQP",
        bounds=[(xl, xu) for xl, xu in zip(set_opt["x_lower"], set_opt["x_upper"])],
        # constraints={"type": "ineq", "fun": mass, "jac": mass_grad},
        options={"disp": 1, "maxiter": 100, "ftol": 1e-6},
        callback=lambda x: print("min mode for midpoint: %f" % obj(x)),
    )

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax = beam.plot_tube(ax, res.x)
    ax.set_aspect("equal")
    plt.savefig("output/eigenvector.pdf")


elif problem == "optimization_eigenvalue":
    # sets for the beam
    set_beam = {
        "nelems": 50,
        "ndvs": 5,
        "L": 1.0,
        "t": 0.025,
        "N": 5,
        "ksrho": 10.0,
    }

    beam = EulerBeam(**set_beam)

    # sets for the optimization
    set_opt = {
        "x0": 0.1 * beam.L * np.ones(beam.ndvs),
        "x_con": 0.15 * beam.L * np.ones(beam.ndvs),
        "x_lower": 0.1 * beam.L * np.ones(beam.ndvs),
        "x_upper": 0.25 * beam.L * np.ones(beam.ndvs),
    }

    # minimize the eigenvalue
    obj = lambda x: beam.appox_min_eigenvalue(x)
    obj_grad = lambda x: beam.approx_min_eigenvalue_deriv(x)

    # constrain the mass
    mass = lambda x: beam.get_mass(set_opt["x_con"]) - beam.get_mass(x)
    mass_grad = lambda x: beam.get_mass_deriv()

    res = minimize(
        obj,
        set_opt["x0"],
        jac=obj_grad,
        method="SLSQP",
        bounds=[(xl, xu) for xl, xu in zip(set_opt["x_lower"], set_opt["x_upper"])],
        constraints={"type": "ineq", "fun": mass, "jac": mass_grad},
        options={"disp": 1, "maxiter": 100, "ftol": 1e-8},
        callback=lambda x: print("min eigenvalue: ", obj(x)),
    )

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax = beam.plot_modes(ax, res.x, n=4, scale=4.0)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.invert_yaxis()
    plt.savefig("output/eigenvalue.pdf")

elif problem == "stress":
    # sets for the beam
    # [50, 50, 1.0, 0.025, 5, 0.01] + [0.01, 0.05, 0.01, 0.1]
    # set_beam = {
    #     "nelems": 500,
    #     "ndvs": 50,
    #     "L": 1.0,
    #     "t": 0.025,
    #     "N": 4,
    #     "ksrho": 0.01,
    # }

    # set the beam for aluminum
    set_beam = {
        "nelems": 10,
        "ndvs": 10,
        "L": 1.0,  # 1 m
        "t": 0.025,  # 2.5 cm
        "N": 4,
        "ksrho": 1.0,
        "E": 70e9,  # 70 GPa
        "density": 2700.0,  # 2700 kg/m^3
    }

    beam = EulerBeam(**set_beam)
    dof = np.arange(0, beam.nelems // 4)
    beam.D[dof, dof] = 1.0

    # sets for the stress
    set_stress = {
        "rho": 100.0,
        "allowable": 276e6,  # 276 MPa
    }

    # sets for the optimization
    set_opt = {
        "x0": 0.05 * beam.L * np.ones(beam.ndvs),
        "x_con": 0.1 * beam.L * np.ones(beam.ndvs),
        "x_lower": 0.025 * beam.L * np.ones(beam.ndvs),
        "x_upper": 0.2 * beam.L * np.ones(beam.ndvs),
        "options": {"disp": 1, "maxiter": 100, "ftol": 1e-8},
    }

    rho = set_stress["rho"]
    allowable = set_stress["allowable"]

    x = 0.1 * np.ones(beam.ndvs)
    p = np.random.uniform(size=x.shape)
    dfdx = beam.eigenvector_stress_deriv(x, rho=rho, allowable=allowable)
    exact = beam.exact_eigenvector_stress_deriv(x, rho=rho, allowable=allowable)
    approx = np.dot(dfdx, p)
    exact = np.dot(exact, p)
    ic(np.abs((exact - approx)) / exact)
    
    dfdx = beam.approx_eigenvector_deriv(x)
    exact = beam.exact_eigenvector_deriv(x)

    approx = np.dot(dfdx, p)
    exact = np.dot(exact, p)
    ic((exact - approx) / exact)
    
    # # check the stress derivative
    # if True:
    #     x = 0.1 * np.ones(beam.ndvs)
    #     p = np.random.uniform(size=x.shape)

    #     exact = beam.exact_eigenvector_stress_deriv(x, rho=rho, allowable=allowable)
    #     dfdx = beam.eigenvector_stress_deriv(x, rho=rho, allowable=allowable)

    #     ic(set_beam["nelems"])
    #     # dfdx = beam.approx_eigenvector_deriv(x)
    #     # exact = np.dot(p, beam.exact_eigenvector_deriv(x))

    #     approx = np.dot(dfdx, p)
    #     exact = np.dot(exact, p)
    #     ic((exact - approx) / exact)

    # for dh in 10 ** np.linspace(-4, -10, 10):
    #     # fd = (
    #     #     beam.exact_eigenvector(x + dh * p) - beam.exact_eigenvector(x - dh * p)
    #     # ) / (2.0 * dh)
    #     fd = (
    #         beam.eigenvector_stress(x + dh * p, rho=rho, allowable=allowable)
    #         - beam.eigenvector_stress(x - dh * p, rho=rho, allowable=allowable)
    #     ) / (2.0 * dh)

    #     # ans2 = np.dot(dfdx2, p)

    #     # print("dh      = %15.5e" % (dh))
    #     # print("fd      = %15.5f" % (fd))
    #     # prin
    #     # t("ans     = %15.5f" % (ans))
    #     # ic(dh)
    #     ic((fd - approx) / fd)
    #     ic((fd - exact) / fd)

    # exit(0)

    obj = lambda x: 1e-3 * beam.eigenvector_stress(x, rho=rho, allowable=allowable)
    obj_grad = lambda x: 1e-3 * beam.eigenvector_stress_deriv(
        x, rho=rho, allowable=allowable
    )

    ic(beam.get_mass(set_opt["x_con"]))
    mass = lambda x: 0.1 * (beam.get_mass(set_opt["x_con"]) - beam.get_mass(x))
    mass_grad = lambda x: 0.1 * beam.get_mass_deriv()

    # minimize the stress
    res = minimize(
        obj,
        set_opt["x0"],
        jac=obj_grad,
        method="SLSQP",
        bounds=[(xl, xu) for xl, xu in zip(set_opt["x_lower"], set_opt["x_upper"])],
        constraints={"type": "ineq", "fun": mass, "jac": mass_grad},
        options=set_opt["options"],
        callback=lambda x: print("stress: ", obj(x)),
    )
    ic(beam.N @ res.x)

    # save and read the result
    np.save("output/stress.npy", res.x)
    res_x = np.load("output/stress.npy")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax = beam.plot_tube(ax, res_x)
    ax.set_aspect("equal")
    plt.savefig("output/stress_tube.pdf")

    with plt.style.context(["science", "nature"]):
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = plt.axes(projection="3d")
        ax = beam.plot_stress(ax, res_x)
        ax = beam.plot_modes(ax, res_x, n=set_beam["N"], scale=2.0)
        ax.set_aspect("equal")
        plt.savefig("output/stress.pdf")


plt.show()
