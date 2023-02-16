import numpy as np
from scipy import sparse
from scipy import spatial
from scipy.sparse import linalg, coo_matrix
import matplotlib.pylab as plt
import matplotlib.tri as tri
from mpi4py import MPI
from paropt import ParOpt
import pickle
import argparse
import os


class NodeFilter:
    """
    A node-based filter for topology optimization
    """

    def __init__(
        self, conn, X, r0=1.0, ftype="spatial", beta=10.0, eta=0.5, projection=False
    ):
        """
        Create a filter
        """
        self.conn = np.array(conn)
        self.X = np.array(X)
        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1

        # Store information about the projection
        self.beta = beta
        self.eta = eta
        self.projection = projection

        # Store information about the filter
        self.F = None
        self.A = None
        self.B = None
        if ftype == "spatial":
            self._initialize_spatial(r0)
        else:
            self._initialize_helmholtz(r0)

        return

    def _initialize_spatial(self, r0):
        """
        Initialize the spatial filter
        """

        # Create a KD tree
        tree = spatial.KDTree(self.X)

        F = sparse.lil_matrix((self.nnodes, self.nnodes))
        for i in range(self.nnodes):
            indices = tree.query_ball_point(self.X[i, :], r0)
            Fhat = np.zeros(len(indices))

            for j, index in enumerate(indices):
                dist = np.sqrt(
                    np.dot(
                        self.X[i, :] - self.X[index, :], self.X[i, :] - self.X[index, :]
                    )
                )
                Fhat[j] = r0 - dist

            Fhat = Fhat / np.sum(Fhat)
            F[i, indices] = Fhat

        self.F = F.tocsr()
        self.FT = self.F.transpose()

        return

    def _initialize_helmholtz(self, r0):
        i = []
        j = []
        for index in range(self.nelems):
            for ii in self.conn[index, :]:
                for jj in self.conn[index, :]:
                    i.append(ii)
                    j.append(jj)

        # Convert the lists into numpy arrays
        i_index = np.array(i, dtype=int)
        j_index = np.array(j, dtype=int)

        # Quadrature points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 4 x 4 element stiffness matrices
        Ae = np.zeros((self.nelems, 4, 4))
        Ce = np.zeros((self.nelems, 4, 4))

        Be = np.zeros((self.nelems, 2, 4))
        He = np.zeros((self.nelems, 1, 4))
        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N = 0.25 * np.array(
                    [
                        (1.0 - xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 + eta),
                        (1.0 - xi) * (1.0 + eta),
                    ]
                )
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1] / detJ
                invJ[:, 0, 1] = -J[:, 0, 1] / detJ
                invJ[:, 1, 0] = -J[:, 1, 0] / detJ
                invJ[:, 1, 1] = J[:, 0, 0] / detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                He[:, 0, :] = N
                Be[:, 0, :] = Nx
                Be[:, 1, :] = Ny

                Ce += np.einsum("n,nij,nil -> njl", detJ, He, He)
                Ae += np.einsum("n,nij,nil -> njl", detJ * r0**2, Be, Be)

        # Finish the computation of the Ae matrices
        Ae += Ce

        A = sparse.coo_matrix((Ae.flatten(), (i_index, j_index)))
        A = A.tocsc()
        self.A = linalg.factorized(A)

        B = sparse.coo_matrix((Ce.flatten(), (i_index, j_index)))
        self.B = B.tocsr()
        self.BT = self.B.transpose()

        return

    def apply(self, x):
        if self.F is not None:
            rho = self.F.dot(x)
        else:
            rho = self.A(self.B.dot(x))

        if self.projection:
            denom = np.tanh(self.beta * self.eta) + np.tanh(
                self.beta * (1.0 - self.eta)
            )
            rho = (
                np.tanh(self.beta * self.eta) + np.tanh(self.beta * (rho - self.eta))
            ) / denom

        return rho

    def applyGradient(self, g, x, rho=None):
        if self.F is not None:
            rho = self.F.dot(x)
        else:
            rho = self.A(self.B.dot(x))

        if self.projection:
            denom = np.tanh(self.beta * self.eta) + np.tanh(
                self.beta * (1.0 - self.eta)
            )
            grad = g * (
                (self.beta / denom) * 1.0 / np.cosh(self.beta * (rho - self.eta)) ** 2
            )
        else:
            grad = g

        if self.F is not None:
            return self.FT.dot(grad)
        else:
            return self.BT.dot(self.A(grad))

    def plot(self, u, ax=None, **kwargs):
        """
        Create a plot
        """

        # Create the triangles
        triangles = np.zeros((2 * self.nelems, 3), dtype=int)
        triangles[: self.nelems, 0] = self.conn[:, 0]
        triangles[: self.nelems, 1] = self.conn[:, 1]
        triangles[: self.nelems, 2] = self.conn[:, 2]

        triangles[self.nelems :, 0] = self.conn[:, 0]
        triangles[self.nelems :, 1] = self.conn[:, 2]
        triangles[self.nelems :, 2] = self.conn[:, 3]

        # Create the triangulation object
        tri_obj = tri.Triangulation(self.X[:, 0], self.X[:, 1], triangles)

        if ax is None:
            fig, ax = plt.subplots()

        # Set the aspect ratio equal
        ax.set_aspect("equal")

        # Create the contour plot
        ax.tricontourf(tri_obj, u, **kwargs)

        return


class TopologyAnalysis:
    def __init__(
        self,
        fltr,
        conn,
        X,
        bcs,
        forces={},
        E=10.0,
        nu=0.3,
        P=3.0,
        density=1.0,
        area_fraction=0.4,
        K0=None,
        M0=None,
    ):

        self.fltr = fltr
        self.conn = np.array(conn)
        self.X = np.array(X)
        self.P = P
        self.density = density

        self.K0 = K0
        self.M0 = M0

        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1
        self.nvars = 2 * self.nnodes

        # Compute the constitutivve matrix
        self.C0 = E * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
        )
        self.C0 *= 1.0 / (1.0 - nu**2)

        self.reduced = self._compute_reduced_variables(self.nvars, bcs)
        self.f = self._compute_forces(self.nvars, forces)

        self.area_gradient = self.eval_area_gradient()
        self.fixed_area = area_fraction * np.sum(self.area_gradient)
        self.area_gradient_rho = np.zeros(self.nnodes)

        for k in range(4):
            np.add.at(self.area_gradient_rho, self.conn[:, k], self.area_gradient)
        self.area_gradient_rho[:] *= 0.25

        # Set up the i-j indices for the matrix - these are the row
        # and column indices in the stiffness matrix
        self.var = np.zeros((self.conn.shape[0], 8), dtype=int)
        self.var[:, ::2] = 2 * self.conn
        self.var[:, 1::2] = 2 * self.conn + 1

        i = []
        j = []
        for index in range(self.nelems):
            for ii in self.var[index, :]:
                for jj in self.var[index, :]:
                    i.append(ii)
                    j.append(jj)

        # Convert the lists into numpy arrays
        self.i = np.array(i, dtype=int)
        self.j = np.array(j, dtype=int)
        return

    def _compute_reduced_variables(self, nvars, bcs):
        """
        Compute the reduced set of variables
        """
        reduced = list(range(nvars))

        # For each node that is in the boundary condition dictionary
        for node in bcs:
            uv_list = bcs[node]

            # For each index in the boundary conditions (corresponding to
            # either a constraint on u and/or constraint on v
            for index in uv_list:
                var = 2 * node + index
                reduced.remove(var)

        return reduced

    def _compute_forces(self, nvars, forces):
        """
        Unpack the dictionary containing the forces
        """
        f = np.zeros(nvars)

        for node in forces:
            f[2 * node] += forces[node][0]
            f[2 * node + 1] += forces[node][1]

        return f

    def set_K0(self, K0):
        self.K0 = K0
        return

    def set_M0(self, M0):
        self.M0 = M0
        return

    def assemble_stiffness_matrix(self, C):
        """
        Assemble the stiffness matrix
        """

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element stiffness matrix
        Ke = np.zeros((self.nelems, 8, 8))
        Be = np.zeros((self.nelems, 3, 8))

        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1] / detJ
                invJ[:, 0, 1] = -J[:, 0, 1] / detJ
                invJ[:, 1, 0] = -J[:, 1, 0] / detJ
                invJ[:, 1, 1] = J[:, 0, 0] / detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                Be[:, 0, ::2] = Nx
                Be[:, 1, 1::2] = Ny
                Be[:, 2, ::2] = Ny
                Be[:, 2, 1::2] = Nx

                # This is a fancy (and fast) way to compute the element matrices
                Ke += np.einsum("n,nij,nik,nkl -> njl", detJ, Be, C, Be)

                # This is a slower way to compute the element matrices
                # for k in range(self.nelems):
                #     Ke[k, :, :] += detJ[k]*np.dot(Be[k, :, :].T, np.dot(self.C[k, :, :], Be[k, :, :]))

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        return K

    def stiffness_matrix_derivative(self, psi, u):
        """
        Compute the derivative of the stiffness matrix times the vectors psi and u
        """
        dfdC = np.zeros((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element stiffness matrix
        Be = np.zeros((self.nelems, 3, 8))

        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        psie = np.zeros((self.nelems, 8))

        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        psie[:, ::2] = psi[2 * self.conn]
        psie[:, 1::2] = psi[2 * self.conn + 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1] / detJ
                invJ[:, 0, 1] = -J[:, 0, 1] / detJ
                invJ[:, 1, 0] = -J[:, 1, 0] / detJ
                invJ[:, 1, 1] = J[:, 0, 0] / detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                Be[:, 0, ::2] = Nx
                Be[:, 1, 1::2] = Ny
                Be[:, 2, ::2] = Ny
                Be[:, 2, 1::2] = Nx

                for k in range(self.nelems):
                    eu = np.dot(Be[k, :], ue[k, :])
                    epsi = np.dot(Be[k, :], psie[k, :])

                    dfdC[k, :, :] += detJ[k] * np.outer(epsi, eu)

        return dfdC

    def assemble_mass_matrix(self, rhoE):
        """
        Assemble the mass matrix
        """

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element mass matrices
        Me = np.zeros((self.nelems, 8, 8))
        He = np.zeros((self.nelems, 2, 8))

        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N = 0.25 * np.array(
                    [
                        (1.0 - xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 + eta),
                        (1.0 - xi) * (1.0 + eta),
                    ]
                )
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]

                # Set the B matrix for each element
                He[:, 0, ::2] = N
                He[:, 1, 1::2] = N

                # This is a fancy (and fast) way to compute the element matrices
                Me += self.density * np.einsum("n,nij,nil -> njl", rhoE * detJ, He, He)

        M = sparse.coo_matrix((Me.flatten(), (self.i, self.j)))
        M = M.tocsr()

        return M

    def mass_matrix_derivative(self, u, v):
        """
        Compute the derivative of the mass matrix
        """

        dfdrhoE = np.zeros(self.nelems)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # The interpolation matrix for each element
        He = np.zeros((self.nelems, 2, 8))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        ve = np.zeros((self.nelems, 8))

        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        ve[:, ::2] = v[2 * self.conn]
        ve[:, 1::2] = v[2 * self.conn + 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                N = 0.25 * np.array(
                    [
                        (1.0 - xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 - eta),
                        (1.0 + xi) * (1.0 + eta),
                        (1.0 - xi) * (1.0 + eta),
                    ]
                )
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]

                # Set the B matrix for each element
                He[:, 0, ::2] = N
                He[:, 1, 1::2] = N

                for k in range(self.nelems):
                    eu = np.dot(He[k, :], ue[k, :])
                    ev = np.dot(He[k, :], ve[k, :])

                    dfdrhoE[k] += self.density * detJ[k] * np.dot(ev, eu)

        return dfdrhoE

    def reduce_vector(self, forces):
        """
        Eliminate essential boundary conditions from the vector
        """
        return forces[self.reduced]

    def reduce_matrix(self, matrix):
        """
        Eliminate essential boundary conditions from the matrix
        """
        temp = matrix[self.reduced, :]
        return temp[:, self.reduced]

    def solve(self, C):
        """
        Perform a linear static analysis
        """

        K = self.assemble_stiffness_matrix(C)
        Kr = self.reduce_matrix(K)
        fr = self.reduce_vector(self.f)

        ur = sparse.linalg.spsolve(Kr, fr)

        u = np.zeros(self.nvars)
        u[self.reduced] = ur

        return u

    def compliance(self, x):
        # Compute the density at each node
        self.rho = self.fltr.apply(x)

        # Average the density to get the element-wise density
        self.rhoE = 0.25 * (
            self.rho[self.conn[:, 0]]
            + self.rho[self.conn[:, 1]]
            + self.rho[self.conn[:, 2]]
            + self.rho[self.conn[:, 3]]
        )

        # Compute the element stiffnesses
        self.C = np.outer(self.rhoE**self.P, self.C0)
        self.C = self.C.reshape((self.nelems, 3, 3))

        self.u = self.solve(self.C)

        return self.f.dot(self.u)

    def compliance_gradient(self, x):
        # Compute the gradient of the compliance
        dfdC = -1.0 * self.stiffness_matrix_derivative(self.u, self.u)

        dfdrhoE = np.zeros(self.nelems)
        for i in range(3):
            for j in range(3):
                dfdrhoE[:] += self.C0[i, j] * dfdC[:, i, j]

        dfdrhoE[:] *= self.P * (self.rhoE) ** (self.P - 1.0)

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return self.fltr.applyGradient(dfdrho, x)

    def eval_area_gradient(self):

        dfdrhoE = np.zeros(self.nelems)

        # Quadrature points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Jacobian transformation
        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]

                dfdrhoE[:] += detJ

        return dfdrhoE

    def frequencies(self, x, k=5, sigma=0.0):
        """
        Compute the k-th smallest natural frequencies
        """

        # Compute the density at each node
        self.rho = self.fltr.apply(x)

        # Average the density to get the element-wise density
        self.rhoE = 0.25 * (
            self.rho[self.conn[:, 0]]
            + self.rho[self.conn[:, 1]]
            + self.rho[self.conn[:, 2]]
            + self.rho[self.conn[:, 3]]
        )

        # Compute the element stiffnesses
        self.C = np.outer(self.rhoE**self.P, self.C0)
        self.C = self.C.reshape((self.nelems, 3, 3))

        K = self.assemble_stiffness_matrix(self.C)
        if self.K0 is not None:
            K += self.K0
        Kr = self.reduce_matrix(K)

        M = self.assemble_mass_matrix(self.rhoE)
        if self.M0 is not None:
            M += self.M0
        Mr = self.reduce_matrix(M)

        # Find the eigenvalues closest to zero. This uses a shift and
        # invert strategy around sigma = 0, which means that the largest
        # magnitude values are closest to zero.
        eigs, ur = sparse.linalg.eigsh(Kr, M=Mr, k=5, sigma=sigma, which="LM", tol=1e-6)

        u = np.zeros((self.nvars, k))
        for i in range(k):
            u[self.reduced, i] = ur[:, i]

        return np.sqrt(eigs), u

    def ks_eigenvalue(self, x, ks_rho=100.0, k=5):
        """
        Compute the ks minimum eigenvalue
        """
        if k > len(self.reduced):
            k = len(self.reduced)

        self.omega, self.phi = self.frequencies(x, k=k)

        c = np.min(self.omega)
        self.eta = np.exp(-ks_rho * (self.omega - c))
        a = np.sum(self.eta)
        ks_min = c - np.log(a) / ks_rho
        self.eta *= 1.0 / a

        return ks_min

    def ks_eigenvalue_derivative(self, x):
        """
        Compute the ks minimum eigenvalue
        """

        dfdC = np.zeros((self.nelems, 3, 3))
        dfdrhoM = np.zeros(self.nelems)

        ks_grad = np.zeros(self.nelems)
        for i in range(len(self.eta)):
            kx = self.stiffness_matrix_derivative(self.phi[:, i], self.phi[:, i])
            dfdC += (self.eta[i] / (2 * self.omega[i])) * kx

            mx = self.mass_matrix_derivative(self.phi[:, i], self.phi[:, i])
            dfdrhoM -= (self.omega[i] ** 2 * self.eta[i] / (2 * self.omega[i])) * mx

        # Complete the derivative w.r.t. the stiffness matrix
        dfdrhoE = np.zeros(self.nelems)
        for i in range(3):
            for j in range(3):
                dfdrhoE[:] += self.C0[i, j] * dfdC[:, i, j]

        dfdrhoE[:] *= self.P * (self.rhoE) ** (self.P - 1.0)

        # Add the derivative w.r.t. the mass matrix
        dfdrhoE += dfdrhoM

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return self.fltr.applyGradient(dfdrho, x)

    def compute_strains(self, u):
        """
        Compute the strains at each quadrature point
        """

        # The strain at each quadrature point
        strain = np.zeros((self.nelems, 4, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        Be = np.zeros((self.nelems, 3, 8))
        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1] / detJ
                invJ[:, 0, 1] = -J[:, 0, 1] / detJ
                invJ[:, 1, 0] = -J[:, 1, 0] / detJ
                invJ[:, 1, 1] = J[:, 0, 0] / detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                Be[:, 0, ::2] = Nx
                Be[:, 1, 1::2] = Ny
                Be[:, 2, ::2] = Ny
                Be[:, 2, 1::2] = Nx

                index = i + 2 * j
                for k in range(self.nelems):
                    strain[k, index, :] = np.dot(Be[k], ue[k, :])

        return strain

    def compute_strain_derivative(self, dfdstrain):
        """
        Compute the strains at each quadrature point
        """

        # The strain at each quadrature point
        dfdu = np.zeros(2 * self.nnodes)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        Be = np.zeros((self.nelems, 3, 8))
        J = np.zeros((self.nelems, 2, 2))
        invJ = np.zeros(J.shape)

        # Compute the x and y coordinates of each element
        xe = self.x[self.conn, 0]
        ye = self.x[self.conn, 1]

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                Nxi = 0.25 * np.array(
                    [-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)]
                )
                Neta = 0.25 * np.array(
                    [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]
                )

                # Compute the Jacobian transformation at each quadrature points
                J[:, 0, 0] = np.dot(xe, Nxi)
                J[:, 1, 0] = np.dot(ye, Nxi)
                J[:, 0, 1] = np.dot(xe, Neta)
                J[:, 1, 1] = np.dot(ye, Neta)

                # Compute the inverse of the Jacobian
                detJ = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
                invJ[:, 0, 0] = J[:, 1, 1] / detJ
                invJ[:, 0, 1] = -J[:, 0, 1] / detJ
                invJ[:, 1, 0] = -J[:, 1, 0] / detJ
                invJ[:, 1, 1] = J[:, 0, 0] / detJ

                # Compute the derivative of the shape functions w.r.t. xi and eta
                # [Nx, Ny] = [Nxi, Neta]*invJ
                Nx = np.outer(invJ[:, 0, 0], Nxi) + np.outer(invJ[:, 1, 0], Neta)
                Ny = np.outer(invJ[:, 0, 1], Nxi) + np.outer(invJ[:, 1, 1], Neta)

                # Set the B matrix for each element
                Be[:, 0, ::2] = Nx
                Be[:, 1, 1::2] = Ny
                Be[:, 2, ::2] = Ny
                Be[:, 2, 1::2] = Nx

                index = i + 2 * j
                for k in range(self.nelems):
                    dfdue = np.dot(Be[k].T, dfdstrain[k, index, :])

                    dfdu[2 * self.conn[k, :]] += dfdue[::2]
                    dfdu[2 * self.conn[k, :] + 1] += dfdue[1::2]

        return dfdu

    def plot(self, u, ax=None, **kwargs):
        """
        Create a plot
        """

        # Create the triangles
        triangles = np.zeros((2 * self.nelems, 3), dtype=int)
        triangles[: self.nelems, 0] = self.conn[:, 0]
        triangles[: self.nelems, 1] = self.conn[:, 1]
        triangles[: self.nelems, 2] = self.conn[:, 2]

        triangles[self.nelems :, 0] = self.conn[:, 0]
        triangles[self.nelems :, 1] = self.conn[:, 2]
        triangles[self.nelems :, 2] = self.conn[:, 3]

        # Create the triangulation object
        tri_obj = tri.Triangulation(self.X[:, 0], self.X[:, 1], triangles)

        if ax is None:
            fig, ax = plt.subplots()

        # Set the aspect ratio equal
        ax.set_aspect("equal")

        # Create the contour plot
        ax.tricontourf(tri_obj, u, **kwargs)

        return


class OptCompliance(ParOpt.Problem):
    """
    Compliance minimization under volume constraint
    """

    def __init__(
        self,
        analysis: TopologyAnalysis,
        vol_frac=0.4,
        draw_history=True,
        draw_every=1,
        prefix="result",
    ):
        self.analysis = analysis
        self.area_gradient = self.analysis.eval_area_gradient()
        self.fixed_area = vol_frac * np.sum(self.area_gradient)

        super().__init__(MPI.COMM_SELF, analysis.nnodes, 1)

        self.draw_history = draw_history
        self.draw_every = draw_every
        self.prefix = prefix

        self.it_counter = 0
        return

    def getVarsAndBounds(self, x, lb, ub):
        """Get the variable values and bounds"""
        lb[:] = 1e-3
        ub[:] = 1.0
        x[:] = 0.95
        return

    def evalObjCon(self, x):
        """
        Return the objective, constraint and fail flag
        """

        fail = 0
        obj = self.analysis.compliance(x[:])
        con = [self.fixed_area - self.analysis.area_gradient.dot(self.analysis.rhoE)]

        if self.draw_history and self.it_counter % self.draw_every == 0:
            fig, ax = plt.subplots()
            self.analysis.plot(self.analysis.rho, ax=ax)
            ax.set_aspect("equal", "box")
            plt.savefig(os.path.join(self.prefix, "%d.png" % self.it_counter))
            plt.close()

        self.it_counter += 1

        return fail, obj, con

    def evalObjConGradient(self, x, g, A):
        """
        Return the objective, constraint and fail flag
        """

        fail = 0
        g[:] = self.analysis.compliance_gradient(x[:])
        A[0][:] = -self.analysis.fltr.applyGradient(
            self.analysis.area_gradient_rho[:], x[:]
        )

        return fail


class OptFrequency(ParOpt.Problem):
    """
    natural frequency maximization under a volume constraint
    """

    def __init__(
        self,
        analysis: TopologyAnalysis,
        non_design_nodes: list,  # indices for nodes whose density is not controlled by the optimizer
        vol_frac=0.4,
        draw_history=True,
        draw_every=1,
        prefix="result",
        dv_mapping=None,  # If provided, optimizer controls reduced design variable xr only
    ):
        self.analysis = analysis
        self.non_design_nodes = non_design_nodes
        self.xfull = np.zeros(self.analysis.nnodes)
        self.xfull[self.non_design_nodes] = 1.0  # Set non-design mass to 0.0
        self.design_nodes = np.ones(len(self.xfull), dtype=bool)
        self.design_nodes[self.non_design_nodes] = False
        self.dv_mapping = dv_mapping

        # Add more non-design constant to matrices
        self.add_mat0(which="M", density=10.0)

        self.area_gradient = self.analysis.eval_area_gradient()
        self.fixed_area = vol_frac * np.sum(self.area_gradient)

        self.ndv = np.sum(self.design_nodes)
        if dv_mapping is not None:
            self.ndv = dv_mapping.shape[1]
        super().__init__(MPI.COMM_SELF, self.ndv, 1)

        self.draw_history = draw_history
        self.draw_every = draw_every
        self.prefix = prefix

        self.it_counter = 0
        return

    def add_mat0(self, which="K", density=1.0):
        assert which == "K" or which == "M"

        x = density * np.ones(self.analysis.nnodes)
        rho = self.analysis.fltr.apply(x)
        rhoE = 0.25 * (
            rho[self.analysis.conn[:, 0]]
            + rho[self.analysis.conn[:, 1]]
            + rho[self.analysis.conn[:, 2]]
            + rho[self.analysis.conn[:, 3]]
        )
        if which == "M":
            M0 = self.analysis.assemble_mass_matrix(rhoE)
            self.analysis.set_M0(M0)
            return

        # Else if which == "K":
        C = np.outer(rhoE, self.analysis.C0).reshape((self.analysis.nelems, 3, 3))
        K0 = self.analysis.assemble_stiffness_matrix(C)
        self.analysis.set_K0(K0)
        return

    def getVarsAndBounds(self, x, lb, ub):
        lb[:] = 1e-3
        ub[:] = 1.0
        x[:] = 0.95
        return

    def evalObjCon(self, x):
        # Populate the nodal variable for analysis
        if self.dv_mapping is not None:
            self.xfull[:] = self.dv_mapping.dot(x)  # x = E*xr
            self.xfull[self.non_design_nodes] = 1.0
        else:
            self.xfull[self.design_nodes] = x[:]

        # Evaluate the maximize natural frequency
        obj = -self.analysis.ks_eigenvalue(self.xfull)

        # Compute constraint value
        con = [self.fixed_area - self.analysis.area_gradient.dot(self.analysis.rhoE)]

        # Draw design
        if self.draw_history and self.it_counter % self.draw_every == 0:
            fig, ax = plt.subplots()
            self.analysis.plot(self.analysis.rho, ax=ax)
            ax.set_aspect("equal", "box")
            plt.savefig(os.path.join(self.prefix, "%d.png" % self.it_counter))
            plt.close()

        self.it_counter += 1

        fail = 0
        return fail, obj, con

    def evalObjConGradient(self, x, g, A):
        if self.dv_mapping is not None:
            # Populate the nodal variable for analysis
            self.xfull[:] = self.dv_mapping.dot(x)  # x = E*xr
            self.xfull[self.non_design_nodes] = 1.0

            # Evaluate objective gradient
            g[:] = -self.dv_mapping.T.dot(
                self.analysis.ks_eigenvalue_derivative(self.xfull)
            )

            # Evaluate constraint gradient
            A[0][:] = -self.dv_mapping.T.dot(
                self.analysis.fltr.applyGradient(
                    self.analysis.area_gradient_rho[:], self.xfull
                )
            )

        else:
            # Populate the nodal variable for analysis
            self.xfull[self.design_nodes] = x[:]

            # Evaluate objective gradient
            g[:] = -self.analysis.ks_eigenvalue_derivative(self.xfull)[
                self.design_nodes
            ]

            # Evaluate constraint gradient
            A[0][:] = -self.analysis.fltr.applyGradient(
                self.analysis.area_gradient_rho[:], self.xfull
            )[self.design_nodes]

        return 0


try:
    import mma4py

    class MMAProblem(mma4py.Problem):
        def __init__(self, prob: OptFrequency) -> None:
            self.prob = prob
            super().__init__(MPI.COMM_SELF, prob.ndv, prob.ndv, 1)
            return

        def getVarsAndBounds(self, x, lb, ub):
            self.prob.getVarsAndBounds(x, lb, ub)
            return

        def evalObjCon(self, x, cons) -> float:
            _fail, _obj, _cons = self.prob.evalObjCon(x)
            cons[0] = -_cons[0]
            return _obj

        def evalObjConGrad(self, x, g, gcon):
            self.prob.evalObjConGradient(x, g, gcon)
            gcon[0, :] = -gcon[0, :]
            return

except:
    MMAProblem = None


def create_cantilever_domain(lx=20, ly=10, m=128, n=64):
    """
    Args:
        lx: x-directional length
        ly: y-directional length
        m: number of elements along x direction
        n: number of elements along y direction
    """

    # Generate the square domain problem by default
    nelems = m * n
    nnodes = (m + 1) * (n + 1)

    y = np.linspace(0, ly, n + 1)
    x = np.linspace(0, lx, m + 1)
    nodes = np.arange(0, (n + 1) * (m + 1)).reshape((n + 1, m + 1))

    # Set the node locations
    X = np.zeros((nnodes, 2))
    for j in range(n + 1):
        for i in range(m + 1):
            X[i + j * (m + 1), 0] = x[i]
            X[i + j * (m + 1), 1] = y[j]

    # Set the connectivity
    conn = np.zeros((nelems, 4), dtype=int)
    for j in range(n):
        for i in range(m):
            conn[i + j * m, 0] = nodes[j, i]
            conn[i + j * m, 1] = nodes[j, i + 1]
            conn[i + j * m, 2] = nodes[j + 1, i + 1]
            conn[i + j * m, 3] = nodes[j + 1, i]

    # Find indices of non-design mass
    non_design_nodes = []
    for j in range((n + 1) // 2 - 10, (n + 1) // 2 + 11):
        for i in range(m + 1 - 10, m + 1):
            non_design_nodes.append(nodes[j, i])

    # Set the constrained degrees of freedom at each node
    bcs = {}
    for j in range(n):
        bcs[nodes[j, 0]] = [0, 1]

    P = 10.0
    forces = {}
    pn = n // 10
    for j in range(pn):
        forces[nodes[j, -1]] = [0, -P / pn]

    r0 = 0.05 * np.min((lx, ly))
    return conn, X, r0, bcs, forces, non_design_nodes


def create_square_domain(l=10, npquarter=30):
    """
    Args:
        l: length of the square
        npquarter: number of elements along each edge
    """

    # Generate the square domain problem by default
    lx = l
    ly = l
    m = 2 * npquarter - 1  # Number of elements in x direction
    n = 2 * npquarter - 1  # Number of elements in y direction

    nelems = m * n
    nnodes = (m + 1) * (n + 1)

    y = np.linspace(0, ly, n + 1)
    x = np.linspace(0, lx, m + 1)
    nodes = np.arange(0, (n + 1) * (m + 1)).reshape((n + 1, m + 1))

    # Set the node locations
    X = np.zeros((nnodes, 2))
    for j in range(n + 1):
        for i in range(m + 1):
            X[i + j * (m + 1), 0] = x[i]
            X[i + j * (m + 1), 1] = y[j]

    # Set the connectivity
    conn = np.zeros((nelems, 4), dtype=int)
    for j in range(n):
        for i in range(m):
            conn[i + j * m, 0] = nodes[j, i]
            conn[i + j * m, 1] = nodes[j, i + 1]
            conn[i + j * m, 2] = nodes[j + 1, i + 1]
            conn[i + j * m, 3] = nodes[j + 1, i]

    # Find indices of non-design mass
    non_design_nodes = []
    offset = int(npquarter / 5)
    for j in range((n + 1) // 2 - offset, (n + 1) // 2 + offset):
        for i in range((m + 1) // 2 - offset, (m + 1) // 2 + offset):
            non_design_nodes.append(nodes[j, i])

    # Constrain all boundaries
    bcs = {}
    for j in range(n + 1):
        bcs[nodes[j, 0]] = [0, 1]
        bcs[nodes[j, m]] = [0, 1]
    for i in range(m + 1):
        bcs[nodes[0, i]] = [0, 1]
        bcs[nodes[n, i]] = [0, 1]

    P = 10.0
    forces = {}
    pn = n // 10
    for j in range(pn):
        forces[nodes[j, -1]] = [0, -P / pn]

    r0 = 0.05 * np.min((lx, ly))

    # Create the mapping E such that x = E*xr, where xr is the nodal variable
    # of a quarter and is controlled by the optimizer, x is the nodal variable
    # of the entire domain
    Ei = []
    Ej = []
    redu_idx = 0
    for j in range((n + 1) // 2):
        for i in range((m + 1) // 2):
            if nodes[j, i] not in non_design_nodes:
                Ej.extend(4 * [redu_idx])
                Ei.extend(
                    [nodes[j, i], nodes[j, m - i], nodes[n - j, i], nodes[n - j, m - i]]
                )
                redu_idx += 1

    Ev = np.ones(len(Ei))
    dv_mapping = coo_matrix((Ev, (Ei, Ej)))

    return conn, X, r0, bcs, forces, non_design_nodes, dv_mapping


def visualize_domain(X, bcs, non_design_nodes=None):
    fig, ax = plt.subplots()

    bc_X = np.array([X[i, :] for i in bcs.keys()])
    ax.scatter(X[:, 0], X[:, 1], color="black")
    ax.scatter(bc_X[:, 0], bc_X[:, 1], color="red")

    if non_design_nodes:
        m0_X = np.array([X[i, :] for i in non_design_nodes])
        ax.scatter(m0_X[:, 0], m0_X[:, 1], color="blue")
    plt.show()
    return


def get_paropt_default_options(prefix, algorithm="tr", maxit=1000):
    options = {
        "algorithm": algorithm,
        "tr_init_size": 0.05,
        "tr_min_size": 1e-6,
        "tr_max_size": 10.0,
        "tr_eta": 0.25,
        "tr_infeas_tol": 1e-6,
        "tr_l1_tol": 1e-3,
        "tr_linfty_tol": 0.0,
        "tr_adaptive_gamma_update": True,
        "tr_max_iterations": maxit,
        "mma_max_iterations": maxit,
        "max_major_iters": 100,
        "penalty_gamma": 1e3,
        "qn_subspace_size": 10,
        "qn_type": "bfgs",
        "abs_res_tol": 1e-8,
        "starting_point_strategy": "affine_step",
        "barrier_strategy": "mehrotra_predictor_corrector",
        "use_line_search": False,
        "output_file": os.path.join(prefix, "paropt.out"),
        "tr_output_file": os.path.join(prefix, "paropt.tr"),
        "mma_output_file": os.path.join(prefix, "paropt.mma"),
    }
    return options


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", default="mma4py", choices=["pmma", "mma4py", "tr"])
    p.add_argument(
        "--npquarter", default=48, type=int, help="number of nodes for half-edge"
    )
    p.add_argument("--maxit", default=1000, type=int)
    p.add_argument("--prefix", default="result", type=str)
    args = p.parse_args()

    if not os.path.isdir(args.prefix):
        os.mkdir(args.prefix)

    # conn, X, r0, bcs, forces, non_design_nodes = create_cantilever_domain()
    conn, X, r0, bcs, forces, non_design_nodes, dv_mapping = create_square_domain(
        npquarter=args.npquarter
    )

    # Create the filter
    fltr = NodeFilter(conn, X, r0, ftype="spatial", projection=False)

    # Create analysis
    analysis = TopologyAnalysis(fltr, conn, X, bcs, forces)

    # Create optimization problem
    topo = OptFrequency(
        analysis,
        non_design_nodes,
        draw_every=5,
        prefix=args.prefix,
        dv_mapping=dv_mapping,
    )

    if args.optimizer == "mma4py":

        if MMAProblem is None:
            raise ImportError("Cannot use mma4py, package not found.")

        mmaprob = MMAProblem(topo)
        mmaopt = mma4py.Optimizer(
            mmaprob, log_name=os.path.join(args.prefix, "mma4py.log")
        )
        mmaopt.checkGradients()
        mmaopt.optimize(niter=args.maxit, verbose=True)

    else:
        topo.checkGradients()

        if args.optimizer == "pmma":
            algorithm = "mma"
        else:
            algorithm = args.optimizer
        options = get_paropt_default_options(
            args.prefix, algorithm=algorithm, maxit=args.maxit
        )

        # Set up the optimizer
        opt = ParOpt.Optimizer(topo, options)

        # Set a new starting point
        opt.optimize()
        x, z, zw, zl, zu = opt.getOptimizedPoint()
