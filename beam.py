import numpy as np
from scipy.linalg import eigh, expm
import matplotlib.pylab as plt
import mpmath as mp


class EulerBeam:
    """
    Optimization code for a clamped-clamped beam
    """

    def __init__(
        self, L, nelems, ndvs=5, E=1.0, density=1.0, inner_radius=0.1, ksrho=10.0
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
        inner_radius : float
            Inner radius of the tube
        """

        self.L = L
        self.nelems = nelems
        self.ndof = 4 * (self.nelems - 1)
        self.ndvs = ndvs
        self.E = E
        self.density = density
        self.inner_radius = inner_radius

        self.ksrho = ksrho
        self.D = np.zeros((self.ndof, self.ndof))

        u = np.linspace(0.5 / self.nelems, 1.0 - 0.5 / self.nelems, self.nelems)
        self.N = self.eval_bernstein(u, self.ndvs)

        # Length of one element
        Le = self.L / self.nelems

        # dof are stored by: v, w, theta,y, theta,z
        # v, theta,z - beam deformation in the y-z plane
        xydof = [0, 3, 4, 7]
        # w, theta,y
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

    def get_vars(self, elem):
        """
        Get the global variables for the given element

        Parameters
        ----------
        elem : int
            Element index

        Returns
        -------
        elem_vars : list
            List of length 8 of the associated element variables
        """
        if elem == 0:
            return [-1, -1, -1, -1, 0, 1, 2, 3]
        elif elem == self.nelems - 1:
            i = 4 * (elem - 1)
            return [i, i + 1, i + 2, i + 3, -1, -1, -1, -1]
        else:
            i = 4 * (elem - 1)
            j = 4 * elem
            return [i, i + 1, i + 2, i + 3, j, j + 1, j + 2, j + 3]

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

        # Compute the thickness
        t = np.dot(self.N, x)

        # Compute the sectional mass
        t0 = t + self.inner_radius
        rhoA = self.density * np.pi * (t0**2 - self.inner_radius**2)

        return rhoA

    def get_sectional_mass_deriv(self, x, dfdrhoA):
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
        """

        # Compute the thickness
        t = np.dot(self.N, x)

        # Compute the sectional mass
        t0 = t + self.inner_radius
        dfdx = 2.0 * self.density * np.pi * np.dot(self.N.T, t0 * dfdrhoA)

        return dfdx

    def get_mass(self, x):
        """
        Get the mass of the beam
        """
        Le = self.L / self.nelems
        return Le * np.sum(self.get_sectional_mass(x))

    def get_mass_deriv(self, x):
        """
        Get the derivative of the mass of the beam
        """
        Le = self.L / self.nelems
        dfdrhoA = Le * np.ones(self.nelems)
        return np.sum(self.get_sectional_mass_deriv(x, dfdrhoA))

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

    def get_mass_matrix_deriv(self, x, u, v):
        dfdrhoA = np.zeros((self.nelems))

        for k in range(self.nelems):
            elem_vars = self.get_vars(k)
            for ie, i in enumerate(elem_vars):
                for je, j in enumerate(elem_vars):
                    if i >= 0 and j >= 0:
                        dfdrhoA[k] += u[i] * v[j] * self.me[ie, je]

        return self.get_sectional_mass_deriv(x, dfdrhoA)

    def get_sectional_stiffness(self, x):
        """
        Given the design variables, compute the sectional stiffness
        """

        # Compute the thickness
        t = np.dot(self.N, x)

        # Compute the sectional bending stiffness
        t0 = t + self.inner_radius
        EI = self.E * np.pi * (t0**4 - self.inner_radius**4)

        return EI

    def get_sectional_stiffness_deriv(self, x, dfdEI):
        """
        Given the design variables, compute the sectional stiffness
        """

        # Compute the thickness
        t = np.dot(self.N, x)

        # Compute the sectional bending stiffness
        t0 = t + self.inner_radius

        dfdx = 4.0 * self.E * np.pi * np.dot(self.N.T, t0**3 * dfdEI)

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

    def solve_eigenvalue_problem(self, x, N=5):

        A = self.get_stiffness_matrix(x)
        B = self.get_mass_matrix(x)

        # Compute the eigenvalues of the generalized eigen problem
        lam, Q = eigh(A, B)

        return lam[:N], Q[:, :N]

    def appox_min_eigenvalue(self, x, N=5):
        lam, Q = self.solve_eigenvalue_problem(x, N=N)

        min_lam = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - min_lam))
        return min_lam - np.log(np.sum(eta)) / self.ksrho

    def approx_min_eigenvalue_deriv(self, x, N=5):
        lam, Q = self.solve_eigenvalue_problem(x, N=N)

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

    def exact_eigenvector(self, x):
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

    def approx_eigenvector(self, x, N=5):

        lam, QN = self.solve_eigenvalue_problem(x, N=N)

        eta = np.exp(-self.ksrho * (lam - np.min(lam)))
        eta = eta / np.sum(eta)

        h = 0.0
        for i in range(N):
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

    def exact_eigvector_deriv(self, x):
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
                    x, Q[:, i], Q[:, j]
                )

        return dfdx

    def approx_eigenvector_deriv(self, x, N=5):
        """
        Approximately compute the forward derivative
        """

        lam, QN = self.solve_eigenvalue_problem(x, N=N)

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
                dfdx += Eij * self.get_stiffness_matrix_deriv(x, QN[:, i], QN[:, j])

                # Add to dfdx from B
                dfdx -= Eij * lam[i] * self.get_mass_matrix_deriv(x, QN[:, i], QN[:, j])

        # Get the stiffness and mass matrices
        A = self.get_stiffness_matrix(x)
        B = self.get_mass_matrix(x)

        # Form the augmented linear system of equations
        for k in range(N):
            # Compute B * vk = D * qk
            bk = np.dot(self.D, QN[:, k])
            vk = np.linalg.solve(B, -eta[k] * bk)
            dfdx += self.get_mass_matrix_deriv(x, QN[:, k], vk)

            # Solve the augmented system of equations for wk
            Ak = A - lam[k] * B
            Ck = np.dot(B, QN)

            # Set up the augmented linear system of equations
            mat = np.block([[Ak, Ck], [Ck.T, np.zeros((N, N))]])
            b = np.zeros(mat.shape[0])

            # Compute the right-hand-side vector
            b[: self.ndof] = -eta[k] * bk

            # Solve the first block linear system of equations
            sol = np.linalg.solve(mat, b)
            wk = sol[: self.ndof]

            # Compute the contributions from the derivative from Adot
            dfdx += 2.0 * self.get_stiffness_matrix_deriv(x, QN[:, k], wk)

            # Add the contributions to the derivative from Bdot here...
            dfdx -= lam[k] * self.get_mass_matrix_deriv(x, QN[:, k], wk)

            # Now, compute the remaining contributions to the derivative from
            # B by solving the second auxiliary system
            dk = np.dot(A, vk)
            b[: self.ndof] = dk

            sol = np.linalg.solve(mat, b)
            uk = sol[: self.ndof]

            # Compute the contributions from the derivative
            dfdx -= self.get_mass_matrix_deriv(x, QN[:, k], uk)

        return dfdx

    def get_stress_values(self, x, eta, Q, allowable=1.0, N=5):

        # Compute the thickness
        t = np.dot(self.N, x)

        # Compute the sectional bending stiffness
        t0 = t + self.inner_radius

        # Loop over all the eigenvalues
        stress = np.zeros(self.nelems)

        for elem in range(self.nelems):
            # Get the element variables
            elem_vars = self.get_vars(elem)

            for k in range(N):
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

    def get_stress_values_deriv(self, x, eta_stress, eta, Q, allowable=1.0, N=5):

        # Compute the thickness
        t = np.dot(self.N, x)

        # Compute the sectional bending stiffness
        t0 = t + self.inner_radius

        dfdt0 = np.zeros(self.nelems)

        for elem in range(self.nelems):
            # Get the element variables
            elem_vars = self.get_vars(elem)

            for k in range(N):
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

        # Compute the thickness
        t = np.dot(self.N, x)

        # Compute the sectional bending stiffness
        t0 = t + self.inner_radius

        product = np.zeros(self.ndof)

        for elem in range(self.nelems):
            # Get the element variables
            elem_vars = self.get_vars(elem)

            # Compute ky = d^2v/dx^2
            ky = np.dot(self.By, Qk[elem_vars])

            # Compute kz = d^2w/dx^2
            kz = np.dot(self.Bz, Qk[elem_vars])

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
                    )

        return product

    def eigenvector_stress(self, x, rho=10.0, allowable=1.0, N=5):
        """
        Compute the aggregated stress value based on the lowest mode shapes

        """

        # Solve the eigenvalue problem
        lam, Q = self.solve_eigenvalue_problem(x, N=N)

        # Compute the eta values
        lam_min = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - lam_min))
        eta = eta / np.sum(eta)

        # Compute the stresses in the beam
        stress = self.get_stress_values(x, eta, Q, allowable=allowable, N=N)

        # Now aggregate over the stress
        max_stress = np.max(stress)
        h = max_stress + np.sum(np.exp(rho * (stress - max_stress))) / rho

        return h

    def eigenvector_stress_deriv(self, x, rho=10.0, allowable=1.0, N=5):

        # Solve the eigenvalue problem
        lam, QN = self.solve_eigenvalue_problem(x, N=N)

        # Compute the eta values
        lam_min = np.min(lam)
        eta = np.exp(-self.ksrho * (lam - lam_min))
        trace = np.sum(eta)
        eta = eta / trace

        # Compute the stresses in the beam
        stress = self.get_stress_values(x, eta, QN, allowable=allowable, N=N)

        # Now aggregate over the stress
        max_stress = np.max(stress)
        eta_stress = np.exp(rho * (stress - max_stress))
        eta_stress = eta_stress / np.sum(eta_stress)

        # Set the value of the derivative
        dfdx = self.get_stress_values_deriv(
            x, eta_stress, eta, QN, allowable=allowable, N=N
        )

        result = np.dot(eta_stress, stress)
        result2 = 0.0

        for k in range(N):
            prod = self.get_stress_product(x, eta_stress, QN[:, k], allowable=allowable)
            result2 += eta[k] * np.dot(QN[:, k], prod)

        print("result = ", result)
        print("result2 = ", result2)

        for j in range(N):
            # Compute D * QN[:, j]
            prod = self.get_stress_product(x, eta_stress, QN[:, j], allowable=allowable)

            for i in range(N):
                qDq = np.dot(QN[:, i], prod)
                scalar = qDq
                if i == j:
                    scalar = qDq - np.dot(eta_stress, stress)

                Eij = scalar * self.precise(self.ksrho, trace, lam_min, lam[i], lam[j])

                # Add to dfdx from A
                dfdx += Eij * self.get_stiffness_matrix_deriv(x, QN[:, i], QN[:, j])

                # Add to dfdx from B
                dfdx -= Eij * lam[i] * self.get_mass_matrix_deriv(x, QN[:, i], QN[:, j])

        # Get the stiffness and mass matrices
        A = self.get_stiffness_matrix(x)
        B = self.get_mass_matrix(x)

        # Form the augmented linear system of equations
        for k in range(N):
            # Compute B * vk = D * qk
            prod = self.get_stress_product(x, eta_stress, QN[:, k], allowable=allowable)
            vk = np.linalg.solve(B, -eta[k] * prod)
            dfdx += self.get_mass_matrix_deriv(x, QN[:, k], vk)

            # Solve the augmented system of equations for wk
            Ak = A - lam[k] * B
            Ck = np.dot(B, QN)

            # Set up the augmented linear system of equations
            mat = np.block([[Ak, Ck], [Ck.T, np.zeros((N, N))]])
            b = np.zeros(mat.shape[0])

            # Compute the right-hand-side vector
            b[: self.ndof] = -eta[k] * prod

            # Solve the first block linear system of equations
            sol = np.linalg.solve(mat, b)
            wk = sol[: self.ndof]

            # Compute the contributions from the derivative from Adot
            dfdx += 2.0 * self.get_stiffness_matrix_deriv(x, QN[:, k], wk)

            # Add the contributions to the derivative from Bdot here...
            dfdx -= lam[k] * self.get_mass_matrix_deriv(x, QN[:, k], wk)

            # Now, compute the remaining contributions to the derivative from
            # B by solving the second auxiliary system
            dk = np.dot(A, vk)
            b[: self.ndof] = dk

            sol = np.linalg.solve(mat, b)
            uk = sol[: self.ndof]

            # Compute the contributions from the derivative
            dfdx -= self.get_mass_matrix_deriv(x, QN[:, k], uk)

        return dfdx

    def plot_modes(self, x, N=5):
        """
        Plot the modes
        """

        u = np.linspace(0.5 / self.nelems, 1.0 - 0.5 / self.nelems, self.nelems)
        xvals = np.dot(self.N, x)

        plt.figure()
        plt.plot(u, xvals)

        lam, QN = self.solve_eigenvalue_problem(x, N=N)

        plt.figure()
        for k in range(N):
            u = np.zeros(self.nelems + 1)
            x = np.linspace(0, self.L, self.nelems + 1)
            u[1:-1] = QN[::4, k]
            plt.plot(x, u)

        plt.show()


np.random.seed(12345)

problem = "stress"


if problem == "exact_derivative":
    ndvs = 5
    L = 1.0
    nelems = 50

    npts = 25
    fd = np.zeros(npts)
    exact = np.zeros(npts)
    approx1 = np.zeros(npts)
    approx2 = np.zeros(npts)
    approx3 = np.zeros(npts)
    approx4 = np.zeros(npts)
    approx5 = np.zeros(npts)
    approx6 = np.zeros(npts)
    approx8 = np.zeros(npts)
    approx10 = np.zeros(npts)

    # Pick a direction
    x = 0.01 * np.ones(ndvs)
    px = np.ones(x.shape)

    # Solve the eigenvalue problem to determine the range of ks values
    beam = EulerBeam(L, nelems, ndvs=ndvs)
    lam, Q = beam.solve_eigenvalue_problem(x)

    ksrho = lam[0] * (10 ** np.linspace(-4, 2, npts))

    for i in range(npts):
        beam = EulerBeam(L, nelems, ndvs=ndvs, ksrho=ksrho[i])

        # Set the matrix component we want to zero
        dof = np.arange(0, nelems // 4)
        beam.D[dof, dof] = 1.0

        dh = 1e-30
        fd[i] = beam.exact_eigenvector(x + 1j * dh * px).imag / dh
        # exact[i] = np.dot(px, beam.exact_eigenvector_deriv(x))
        approx1[i] = np.dot(px, beam.approx_eigenvector_deriv(x, N=1))
        approx2[i] = np.dot(px, beam.approx_eigenvector_deriv(x, N=2))
        approx3[i] = np.dot(px, beam.approx_eigenvector_deriv(x, N=3))
        approx4[i] = np.dot(px, beam.approx_eigenvector_deriv(x, N=4))
        approx5[i] = np.dot(px, beam.approx_eigenvector_deriv(x, N=5))
        approx6[i] = np.dot(px, beam.approx_eigenvector_deriv(x, N=6))
        approx8[i] = np.dot(px, beam.approx_eigenvector_deriv(x, N=8))
        approx10[i] = np.dot(px, beam.approx_eigenvector_deriv(x, N=10))
        print(".", end="", flush=True)
    print("")

    plt.figure()
    plt.semilogx(ksrho, fd, label="complex step")
    # plt.semilogx(ksrho, exact, label="exact")
    plt.semilogx(ksrho, approx1, label="approx N = 1")
    plt.semilogx(ksrho, approx2, label="approx N = 2")
    plt.semilogx(ksrho, approx3, label="approx N = 3")
    plt.semilogx(ksrho, approx4, label="approx N = 4")
    plt.semilogx(ksrho, approx5, label="approx N = 5")
    plt.semilogx(ksrho, approx6, label="approx N = 6")
    plt.semilogx(ksrho, approx8, label="approx N = 8")
    plt.semilogx(ksrho, approx10, label="approx N = 10")
    plt.legend()

    plt.figure()
    # plt.loglog(ksrho, (fd - exact) / fd, label="exact")
    plt.loglog(ksrho, np.abs((fd - approx1) / fd), label="approx N = 1")
    plt.loglog(ksrho, np.abs((fd - approx2) / fd), label="approx N = 2")
    plt.loglog(ksrho, np.abs((fd - approx3) / fd), label="approx N = 3")
    plt.loglog(ksrho, np.abs((fd - approx4) / fd), label="approx N = 4")
    plt.loglog(ksrho, np.abs((fd - approx5) / fd), label="approx N = 5")
    plt.loglog(ksrho, np.abs((fd - approx6) / fd), label="approx N = 6")
    plt.loglog(ksrho, np.abs((fd - approx8) / fd), label="approx N = 8")
    plt.loglog(ksrho, np.abs((fd - approx10) / fd), label="approx N = 10")

    plt.legend()

    plt.show()

elif problem == "stress":
    ndvs = 5
    L = 1.0
    nelems = 50

    # Solve the eigenvalue problem to determine the range of ks values
    beam = EulerBeam(L, nelems, ndvs=ndvs)

    # Set the design variable values
    x = 0.01 * np.ones(ndvs)

    N = 8
    rho = 10.0
    allowable = 1.0

    p = np.random.uniform(size=x.shape)

    dh = 1e-5

    fd = (
        beam.eigenvector_stress(x + dh * p, rho=rho, allowable=allowable)
        - beam.eigenvector_stress(x - dh, rho=rho, allowable=allowable)
    ) / (2.0 * dh)

    dfdx = beam.eigenvector_stress_deriv(x, rho=rho, allowable=allowable, N=N)
    ans = np.dot(dfdx, p)

    print(fd)
    print(ans)
    print(dfdx)


elif problem == "optimization":

    ndvs = 10
    L = 1.0
    nelems = 51
    beam = EulerBeam(L, nelems, ndvs=ndvs)

    x = 0.01 * np.ones(ndvs)

    # from scipy.optimize import minimize

    # obj = lambda x: 0.1 * beam.approx_eigenvector(x)  # >= 0.0
    # obj_grad = lambda x: 0.1 * beam.approx_eigenvector_deriv(x)

    # # obj = lambda x: -0.01 * beam.appox_min_eigenvalue(x)
    # # obj_grad = lambda x: -0.01 * beam.approx_min_eigenvalue_deriv(x)

    # con_mass = lambda x: 0.5 * beam.nelems - np.sum(np.dot(beam.N, x))
    # con_mass_grad = lambda x: -np.sum(beam.N, axis=0)

    # x = np.random.uniform(size=ndvs)
    # px = np.random.uniform(size=ndvs)

    # # dh = 1e-5
    # # gx = con_mass_grad(x)
    # # fd = (con_mass(x + dh * px) - con_mass(x)) / dh
    # # ans = np.dot(gx, px)
    # # print("fd  = ", fd)
    # # print("ans = ", ans)

    # res = minimize(
    #     obj,
    #     x0,
    #     jac=obj_grad,
    #     method="SLSQP",
    #     bounds=[(1e-3, 10.0)] * len(x0),
    #     constraints=[
    #         {"type": "ineq", "fun": con_mass, "jac": con_mass_grad},
    #         # {
    #         #     "type": "eq",
    #         #     "fun": con_mass,
    #         #     "jac": con_mass_grad,
    #         # },
    #     ],
    # )

    # print(res)

    # fd = (beam.appox_min_eigenvalue(x + dh * px) - beam.appox_min_eigenvalue(x)) / dh
    # ans_approx = np.dot(px, beam.approx_min_eigenvalue_deriv(x))

    # print("fd = ", fd)
    # print("ans_approx = ", ans_approx)
