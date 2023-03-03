import numpy as np
from scipy import sparse
from scipy import spatial
from scipy.linalg import eigh
from scipy.sparse import linalg, coo_matrix
import matplotlib.pylab as plt
import matplotlib.tri as tri
from paropt import ParOpt
import argparse
import os
import mpmath as mp
from timeit import default_timer as timer
from utils import time_this, timer_set_threshold


def _populate_Be(nelems, xi, eta, xe, ye, Be):
    """
    Populate B matrices for all elements at a quadrature point
    """
    J = np.zeros((nelems, 2, 2))
    invJ = np.zeros(J.shape)

    Nxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    Neta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

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

    return detJ


def _populate_Be_single(xi, eta, xe, ye, Be):
    """
    Populate B matrix for a single element at a quadrature point
    """
    J = np.zeros((2, 2))
    invJ = np.zeros(J.shape)

    Nxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    Neta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

    # Compute the Jacobian transformation at each quadrature points
    J[0, 0] = np.dot(xe, Nxi)
    J[1, 0] = np.dot(ye, Nxi)
    J[0, 1] = np.dot(xe, Neta)
    J[1, 1] = np.dot(ye, Neta)

    # Compute the inverse of the Jacobian
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    invJ = np.linalg.inv(J)

    # Compute the derivative of the shape functions w.r.t. xi and eta
    # [Nx, Ny] = [Nxi, Neta]*invJ
    Nx = np.outer(invJ[0, 0], Nxi) + np.outer(invJ[1, 0], Neta)
    Ny = np.outer(invJ[0, 1], Nxi) + np.outer(invJ[1, 1], Neta)

    # Set the B matrix for each element
    Be[0, ::2] = Nx
    Be[1, 1::2] = Ny
    Be[2, ::2] = Ny
    Be[2, 1::2] = Nx

    return detJ


def _populate_He_single(xi, eta, xe, ye, He):
    """
    Populate a single H matrix at a quadrature point
    """
    J = np.zeros((2, 2))

    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    Nxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    Neta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

    # Compute the Jacobian transformation at each quadrature points
    J[0, 0] = np.dot(xe, Nxi)
    J[1, 0] = np.dot(ye, Nxi)
    J[0, 1] = np.dot(xe, Neta)
    J[1, 1] = np.dot(ye, Neta)

    # Compute the inverse of the Jacobian
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

    # Set the B matrix for each element
    He[0, ::2] = N
    He[1, 1::2] = N

    return detJ


def _populate_He(nelems, xi, eta, xe, ye, He):
    """
    Populate H matrices for all elements at a quadrature point
    """
    J = np.zeros((nelems, 2, 2))

    N = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    Nxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    Neta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])

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

    return detJ


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
        if self.projection:
            if self.F is not None:
                rho = self.F.dot(x)
            else:
                rho = self.A(self.B.dot(x))

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
        ptype="RAMP",
        p=5.0,
        density=1.0,
        epsilon=0.3,
        K0=None,
        M0=None,
        assume_same_element=False,
    ):

        self.ptype = ptype.lower()
        assert self.ptype == "ramp" or self.ptype == "simp"

        self.fltr = fltr
        self.conn = np.array(conn)
        self.X = np.array(X)
        self.p = p
        self.density = density
        self.epsilon = epsilon

        self.D_index = 23

        self.K0 = K0
        self.M0 = M0
        self.assume_same_element = assume_same_element

        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1
        self.nvars = 2 * self.nnodes

        self.Q = None
        self.eigs = None

        # Compute the constitutivve matrix
        self.C0 = E * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
        )
        self.C0 *= 1.0 / (1.0 - nu**2)

        self.reduced = self._compute_reduced_variables(self.nvars, bcs)
        self.f = self._compute_forces(self.nvars, forces)

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

    @time_this
    def assemble_stiffness_matrix(self, rho):
        """
        Assemble the stiffness matrix
        """

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        # Compute the element stiffnesses
        if self.ptype == "simp":
            C = np.outer(rhoE**self.p, self.C0)
        else:
            C = np.outer(rhoE / (1.0 + self.p * (1.0 - rhoE)), self.C0)
        C = C.reshape((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element stiffness matrix
        Ke = np.zeros((self.nelems, 8, 8))

        if self.assume_same_element:
            Be_ = np.zeros((3, 8))

            # Compute the x and y coordinates of the first element
            xe_ = self.X[self.conn[0], 0]
            ye_ = self.X[self.conn[0], 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_Be_single(xi, eta, xe_, ye_, Be_)

                    # This is a fancy (and fast) way to compute the element matrices
                    Ke += detJ * np.einsum("ij,nik,kl -> njl", Be_, C, Be_)

        else:
            Be = np.zeros((self.nelems, 3, 8))

            # Compute the x and y coordinates of each element
            xe = self.X[self.conn, 0]
            ye = self.X[self.conn, 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_Be(self.nelems, xi, eta, xe, ye, Be)

                    # This is a fancy (and fast) way to compute the element matrices
                    Ke += np.einsum("n,nij,nik,nkl -> njl", detJ, Be, C, Be)

        K = sparse.coo_matrix((Ke.flatten(), (self.i, self.j)))
        K = K.tocsr()

        if self.K0 is not None:
            K += self.K0

        return K

    @time_this
    def stiffness_matrix_derivative(self, rho, psi, u):
        """
        Compute the derivative of the stiffness matrix times the vectors psi and u
        """

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        dfdC = np.zeros((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        psie = np.zeros((self.nelems, 8))

        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        psie[:, ::2] = psi[2 * self.conn]
        psie[:, 1::2] = psi[2 * self.conn + 1]

        if self.assume_same_element:
            Be_ = np.zeros((3, 8))
            xe_ = self.X[self.conn[0], 0]
            ye_ = self.X[self.conn[0], 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_Be_single(xi, eta, xe_, ye_, Be_)

                    dfdC += detJ * np.einsum("im,jl,nm,nl -> nij", Be_, Be_, psie, ue)

        else:
            Be = np.zeros((self.nelems, 3, 8))
            xe = self.X[self.conn, 0]
            ye = self.X[self.conn, 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_Be(self.nelems, xi, eta, xe, ye, Be)

                    dfdC += np.einsum("n,nim,njl,nm,nl -> nij", detJ, Be, Be, psie, ue)

        dfdrhoE = np.zeros(self.nelems)
        for i in range(3):
            for j in range(3):
                dfdrhoE[:] += self.C0[i, j] * dfdC[:, i, j]

        if self.p == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1.0)
        else:
            dfdrhoE[:] *= (1.0 + self.p) / (1.0 + self.p * (1.0 - rhoE)) ** 2

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return dfdrho

    @time_this
    def assemble_mass_matrix(self, rho):
        """
        Assemble the mass matrix
        """

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        # Compute the element density
        if self.ptype == "simp":
            density = self.density * rhoE ** (1.0 / self.p)
        else:
            density = self.density * (self.p + 1.0) * rhoE / (1 + self.p * rhoE)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element mass matrices
        Me = np.zeros((self.nelems, 8, 8))

        if self.assume_same_element:
            He_ = np.zeros((2, 8))

            # Compute the x and y coordinates of each element
            xe_ = self.X[self.conn[0], 0]
            ye_ = self.X[self.conn[0], 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_He_single(xi, eta, xe_, ye_, He_)

                    # This is a fancy (and fast) way to compute the element matrices
                    Me += np.einsum("n,ij,il -> njl", density * detJ, He_, He_)

        else:

            He = np.zeros((self.nelems, 2, 8))

            # Compute the x and y coordinates of each element
            xe = self.X[self.conn, 0]
            ye = self.X[self.conn, 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_He(self.nelems, xi, eta, xe, ye, He)

                    # This is a fancy (and fast) way to compute the element matrices
                    Me += np.einsum("n,nij,nil -> njl", density * detJ, He, He)

        M = sparse.coo_matrix((Me.flatten(), (self.i, self.j)))
        M = M.tocsr()

        if self.M0 is not None:
            M += self.M0

        return M

    @time_this
    def mass_matrix_derivative(self, rho, u, v):
        """
        Compute the derivative of the mass matrix

        """

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        # Derivative with respect to element density
        dfdrhoE = np.zeros(self.nelems)

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        ve = np.zeros((self.nelems, 8))

        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        ve[:, ::2] = v[2 * self.conn]
        ve[:, 1::2] = v[2 * self.conn + 1]

        if self.assume_same_element:
            # The interpolation matrix for each element
            He_ = np.zeros((2, 8))

            # Compute the x and y coordinates of each element
            xe_ = self.X[self.conn[0], 0]
            ye_ = self.X[self.conn[0], 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_He_single(xi, eta, xe_, ye_, He_)

                    eu_ = np.einsum("ij,nj -> ni", He_, ue)
                    ev_ = np.einsum("ij,nj -> ni", He_, ve)
                    dfdrhoE += detJ * np.einsum("ni,ni -> n", eu_, ev_)

        else:
            # The interpolation matrix for each element
            He = np.zeros((self.nelems, 2, 8))

            # Compute the x and y coordinates of each element
            xe = self.X[self.conn, 0]
            ye = self.X[self.conn, 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_He(self.nelems, xi, eta, xe, ye, He)

                    eu = np.einsum("nij,nj -> ni", He, ue)
                    ev = np.einsum("nij,nj -> ni", He, ve)
                    dfdrhoE += np.einsum("n,ni,ni -> n", detJ, eu, ev)

        if self.ptype == "simp":
            dfdrhoE[:] *= self.density * rhoE ** (1.0 / self.p - 1.0) / self.p
        else:
            dfdrhoE[:] *= self.density * (1.0 + self.p) / (1.0 + self.p * rhoE) ** 2

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return dfdrho

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

    def full_vector(self, vec):
        """
        Transform from a reduced vector without dirichlet BCs to the full vector
        """
        temp = np.zeros(self.nvars)
        temp[self.reduced] = vec[:]
        return temp

    def solve(self, x):
        """
        Perform a linear static analysis
        """

        # Compute the density at each node
        rho = self.fltr.apply(x)

        K = self.assemble_stiffness_matrix(rho)
        Kr = self.reduce_matrix(K)
        fr = self.reduce_vector(self.f)

        ur = sparse.linalg.spsolve(Kr, fr)
        u = self.full_vector(ur)

        return u

    def compliance(self, x):
        self.u = self.solve(x)
        return self.f.dot(self.u)

    def compliance_gradient(self, x):
        rho = self.fltr.apply(x)
        dfdrho = -1.0 * self.stiffness_matrix_derivative(rho, self.u, self.u)
        return self.fltr.applyGradient(dfdrho, x)

    def eval_area(self, x):
        rho = self.fltr.apply(x)

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        # Quadrature points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Jacobian transformation
        J = np.zeros((self.nelems, 2, 2))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        area = 0.0
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

                area += np.sum(detJ * rhoE)

        return area

    def eval_area_gradient(self, x):

        dfdrhoE = np.zeros(self.nelems)

        # Quadrature points
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Jacobian transformation
        J = np.zeros((self.nelems, 2, 2))

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

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return self.fltr.applyGradient(dfdrho, x)

    @time_this
    def solve_eigenvalue_problem(self, x, k=5, sigma=0.0, nodal_sols=None):
        """
        Compute the k-th smallest natural frequencies
        Populate nodal_sols and cell_sols if provided
        """

        if k > len(self.reduced):
            k = len(self.reduced)

        # Compute the density at each node
        rho = self.fltr.apply(x)

        K = self.assemble_stiffness_matrix(rho)
        Kr = self.reduce_matrix(K)

        M = self.assemble_mass_matrix(rho)
        Mr = self.reduce_matrix(M)

        # Find the eigenvalues closest to zero. This uses a shift and
        # invert strategy around sigma = 0, which means that the largest
        # magnitude values are closest to zero.
        if k == len(self.reduced):
            eigs, Qr = eigh(Kr.todense(), Mr.todense())
        else:
            eigs, Qr = sparse.linalg.eigsh(
                Kr, M=Mr, k=k, sigma=sigma, which="LM", tol=1e-10
            )

        Q = np.zeros((self.nvars, k))
        for i in range(k):
            Q[self.reduced, i] = Qr[:, i]

        # Save vtk output data
        if nodal_sols is not None:
            nodal_sols["x"] = np.array(x)
            nodal_sols["rho"] = np.array(rho)
            for i in range(k):
                nodal_sols["u%d" % i] = Q[0::2, i]
                nodal_sols["v%d" % i] = Q[1::2, i]

        # Save the eigenvalues and eigenvectors
        self.eigs = eigs
        self.Q = Q

        # Return natural frequencies
        omega = np.sqrt(self.eigs)
        return omega

    def ks_omega(self, ks_rho=100.0):
        """
        Compute the ks minimum eigenvalue
        """

        omega = np.sqrt(self.eigs)

        c = np.min(omega)
        eta = np.exp(-ks_rho * (omega - c))
        a = np.sum(eta)
        ks_min = c - np.log(a) / ks_rho
        eta *= 1.0 / a

        return ks_min

    def ks_omega_derivative(self, x, ks_rho=100.0):
        """
        Compute the ks minimum eigenvalue
        """

        # Compute the density at each node
        rho = self.fltr.apply(x)

        omega = np.sqrt(self.eigs)

        c = np.min(omega)
        eta = np.exp(-ks_rho * (omega - c))
        a = np.sum(eta)
        eta *= 1.0 / a

        dfdrho = np.zeros(self.nnodes)

        for i in range(len(self.eigs)):
            kx = self.stiffness_matrix_derivative(rho, self.Q[:, i], self.Q[:, i])
            dfdrho += (eta[i] / (2 * omega[i])) * kx

            mx = self.mass_matrix_derivative(rho, self.Q[:, i], self.Q[:, i])
            dfdrho -= (omega[i] ** 2 * eta[i] / (2 * omega[i])) * mx

        return self.fltr.applyGradient(dfdrho, x)

    @time_this
    def eigenvector_displacement(self, ks_rho=100.0):

        N = len(self.eigs)
        c = np.min(self.eigs)
        eta = np.exp(-ks_rho * (self.eigs - c))
        a = np.sum(eta)
        eta = eta / a

        h = 0.0
        for i in range(N):
            h += eta[i] * self.Q[self.D_index, i] ** 2

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

    @time_this
    def eigenvector_displacement_deriv(self, x, ks_rho=100.0):
        """
        Approximately compute the forward derivative
        """

        # Compute the filtered variables
        rho = self.fltr.apply(x)

        N = len(self.eigs)
        c = np.min(self.eigs)
        eta = np.exp(-ks_rho * (self.eigs - c))
        a = np.sum(eta)
        eta = eta / a

        # Compute the h values
        h = 0.0
        for i in range(N):
            h += eta[i] * self.Q[self.D_index, i] ** 2

        # Set the value of the derivative
        dfdrho = np.zeros(self.nnodes)

        for j in range(N):
            for i in range(j + 1):
                qDq = self.Q[self.D_index, i] * self.Q[self.D_index, j]
                scalar = qDq
                if i == j:
                    scalar = qDq - h

                Eij = scalar * self.precise(ks_rho, a, c, self.eigs[i], self.eigs[j])

                if i == j:
                    # Add to dfdx from A
                    dfdrho += Eij * self.stiffness_matrix_derivative(
                        rho, self.Q[:, i], self.Q[:, j]
                    )

                    # Add to dfdx from B
                    dfdrho -= (
                        Eij
                        * self.eigs[i]
                        * self.mass_matrix_derivative(rho, self.Q[:, i], self.Q[:, j])
                    )
                else:
                    # Add to dfdx from A
                    dfdrho += (
                        2.0
                        * Eij
                        * self.stiffness_matrix_derivative(
                            rho, self.Q[:, i], self.Q[:, j]
                        )
                    )

                    # Add to dfdx from B
                    dfdrho -= (
                        Eij
                        * (self.eigs[i] + self.eigs[j])
                        * self.mass_matrix_derivative(rho, self.Q[:, i], self.Q[:, j])
                    )

        # Get the stiffness and mass matrices
        A = self.assemble_stiffness_matrix(rho)
        B = self.assemble_mass_matrix(rho)

        Ar = self.reduce_matrix(A)
        Br = self.reduce_matrix(B)
        C = B.dot(self.Q)

        nr = len(self.reduced)
        Cr = np.zeros((nr, N))
        for k in range(N):
            Cr[:, k] = self.reduce_vector(C[:, k])
        Ur, R = np.linalg.qr(Cr)

        # Factorize the mass matrix
        Br = Br.tocsc()
        Bfact = linalg.factorized(Br)

        # Form a full factorization for the preconditioner
        factor = 0.99  # Should always be < 1 to ensure P is positive definite.
        # Make this a parameter we can set??
        P = Ar - factor * self.eigs[0] * Br
        P = P.tocsc()
        Pfactor = linalg.factorized(P)

        def preconditioner(x):
            y = Pfactor(x)
            t = np.dot(Ur.T, y)
            y = y - np.dot(Ur, t)
            return y

        preop = linalg.LinearOperator((nr, nr), preconditioner)

        # Form the augmented linear system of equations
        for k in range(N):
            # Compute B * vk = D * qk
            bk = np.zeros(self.nvars)
            bk[self.D_index] = self.Q[self.D_index, k]
            bkr = -eta[k] * self.reduce_vector(bk)

            vkr = Bfact(bkr)
            vk = self.full_vector(vkr)
            dfdrho += self.mass_matrix_derivative(rho, self.Q[:, k], vk)

            # Form the matrix
            def matrix(x):
                y = Ar.dot(x) - self.eigs[k] * Br.dot(x)
                t = np.dot(Ur.T, y)
                y = y - np.dot(Ur, t)
                return y

            matop = linalg.LinearOperator((nr, nr), matrix)

            # Solve the augmented system of equations for wk
            t = np.dot(Ur.T, bkr)
            bkr = bkr - np.dot(Ur, t)
            wkr, info = linalg.gmres(matop, bkr, M=preop, atol=1e-15, tol=1e-10)
            wk = self.full_vector(wkr)

            # Compute the contributions from the derivative from Adot
            dfdrho += 2.0 * self.stiffness_matrix_derivative(rho, self.Q[:, k], wk)

            # Add the contributions to the derivative from Bdot here...
            dfdrho -= self.eigs[k] * self.mass_matrix_derivative(rho, self.Q[:, k], wk)

            # Now, compute the remaining contributions to the derivative from
            # B by solving the second auxiliary system
            dkr = Ar.dot(vkr)

            t = np.dot(Ur.T, dkr)
            dkr = dkr - np.dot(Ur, t)
            ukr, info = linalg.gmres(matop, dkr, M=preop, atol=1e-15, tol=1e-10)
            uk = self.full_vector(ukr)

            # Compute the contributions from the derivative
            dfdrho -= self.mass_matrix_derivative(rho, self.Q[:, k], uk)

        return self.fltr.applyGradient(dfdrho, x)

    @time_this
    def get_stress_values(self, rho, eta, Q, allowable=1.0):
        """
        Compute the stress at each element
        """

        # Loop over all the eigenvalues
        stress = np.zeros(self.nelems)

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        # Compute the stress relaxation factor
        relax = rhoE / (rhoE + self.epsilon * (1.0 - rhoE))

        if self.assume_same_element:
            Be_ = np.zeros((3, 8))

            # Compute the x and y coordinates of each element
            xe_ = self.X[self.conn[0], 0]
            ye_ = self.X[self.conn[0], 1]

            # Compute the stress in the middle of the element
            xi = 0.0
            eta_ = 0.0
            _populate_Be_single(xi, eta_, xe_, ye_, Be_)

            for k in range(len(eta)):
                qe = np.zeros((self.nelems, 8))
                qe[:, ::2] = Q[2 * self.conn, k]
                qe[:, 1::2] = Q[2 * self.conn + 1, k]

                # Compute the stresses in each element
                s = np.einsum("ij,jk,nk -> ni", self.C0, Be_, qe)

                # Add the contributions from the von Mises stress
                stress += (
                    eta[k]
                    * relax
                    * (
                        s[:, 0] ** 2
                        + s[:, 1] ** 2
                        - s[:, 0] * s[:, 1]
                        + 3.0 * s[:, 2] ** 2
                    )
                ) / allowable**2

        else:
            Be = np.zeros((self.nelems, 3, 8))

            # Compute the x and y coordinates of each element
            xe = self.X[self.conn, 0]
            ye = self.X[self.conn, 1]

            # Compute the stress in the middle of the element
            xi = 0.0
            eta_ = 0.0
            _populate_Be(self.nelems, xi, eta_, xe, ye, Be)

            for k in range(len(eta)):
                qe = np.zeros((self.nelems, 8))
                qe[:, ::2] = Q[2 * self.conn, k]
                qe[:, 1::2] = Q[2 * self.conn + 1, k]

                # Compute the stresses in each element
                s = np.einsum("ij,njk,nk -> ni", self.C0, Be, qe)

                # Add the contributions from the von Mises stress
                stress += (
                    eta[k]
                    * relax
                    * (
                        s[:, 0] ** 2
                        + s[:, 1] ** 2
                        - s[:, 0] * s[:, 1]
                        + 3.0 * s[:, 2] ** 2
                    )
                ) / allowable**2

        return stress

    @time_this
    def eigenvector_stress(self, x, ks_rho=100.0, allowable=1.0, cell_sols=None):

        # Compute the filtered variables
        rho = self.fltr.apply(x)

        # Evaluate eigenvalue eta
        c = np.min(self.eigs)
        eta = np.exp(-ks_rho * (self.eigs - c))
        a = np.sum(eta)
        eta = eta / a

        # Compute the stress values
        stress = self.get_stress_values(rho, eta, self.Q, allowable=allowable)

        if cell_sols is not None:
            cell_sols["stress"] = stress

        # Now aggregate over the stress
        max_stress = np.max(stress)
        h = max_stress + np.sum(np.exp(ks_rho * (stress - max_stress))) / ks_rho

        return h

    @time_this
    def eigenvector_stress_derivative(self, x, ks_rho=100.0, allowable=1.0):

        # Compute the filtered variables
        rho = self.fltr.apply(x)

        N = len(self.eigs)
        c = np.min(self.eigs)
        eta = np.exp(-ks_rho * (self.eigs - c))
        a = np.sum(eta)
        eta = eta / a

        # Compute the stress values
        stress = self.get_stress_values(rho, eta, self.Q, allowable=allowable)

        # Now aggregate over the stress
        max_stress = np.max(stress)
        eta_stress = np.exp(ks_rho * (stress - max_stress))
        eta_stress = eta_stress / np.sum(eta_stress)

        # Set the value of the derivative
        dfdrho = self.get_stress_values_deriv(
            rho, eta_stress, eta, self.Q, allowable=allowable
        )

        for j in range(N):
            # Compute D * Q[:, j]
            prod = self.get_stress_product(
                rho, eta_stress, self.Q[:, j], allowable=allowable
            )

            for i in range(j + 1):
                qDq = np.dot(self.Q[:, i], prod)
                scalar = qDq
                if i == j:
                    scalar = qDq - np.dot(eta_stress, stress)

                Eij = scalar * self.precise(ks_rho, a, c, self.eigs[i], self.eigs[j])

                if i == j:
                    # Add to dfdx from A
                    dfdrho += Eij * self.stiffness_matrix_derivative(
                        rho, self.Q[:, i], self.Q[:, j]
                    )

                    # Add to dfdx from B
                    dfdrho -= (
                        Eij
                        * self.eigs[i]
                        * self.mass_matrix_derivative(rho, self.Q[:, i], self.Q[:, j])
                    )
                else:
                    # Add to dfdx from A
                    dfdrho += (
                        2.0
                        * Eij
                        * self.stiffness_matrix_derivative(
                            rho, self.Q[:, i], self.Q[:, j]
                        )
                    )

                    # Add to dfdx from B
                    dfdrho -= (
                        Eij
                        * (self.eigs[i] + self.eigs[j])
                        * self.mass_matrix_derivative(rho, self.Q[:, i], self.Q[:, j])
                    )

        # Get the stiffness and mass matrices
        A = self.assemble_stiffness_matrix(rho)
        B = self.assemble_mass_matrix(rho)

        Ar = self.reduce_matrix(A)
        Br = self.reduce_matrix(B)
        C = B.dot(self.Q)

        nr = len(self.reduced)
        Cr = np.zeros((nr, N))
        for k in range(N):
            Cr[:, k] = self.reduce_vector(C[:, k])
        Ur, R = np.linalg.qr(Cr)

        # Factorize the mass matrix
        Br = Br.tocsc()
        Bfact = linalg.factorized(Br)

        # Form a full factorization for the preconditioner
        factor = 0.99  # Should always be < 1 to ensure P is positive definite.
        # Make this a parameter we can set??
        P = Ar - factor * self.eigs[0] * Br
        P = P.tocsc()
        Pfactor = linalg.factorized(P)

        def preconditioner(x):
            y = Pfactor(x)
            t = np.dot(Ur.T, y)
            y = y - np.dot(Ur, t)
            return y

        preop = linalg.LinearOperator((nr, nr), preconditioner)

        # Form the augmented linear system of equations
        for k in range(N):
            # Compute B * vk = D * qk
            bk = self.get_stress_product(rho, eta_stress, self.Q[:, k])
            bkr = -eta[k] * self.reduce_vector(bk)

            vkr = Bfact(bkr)
            vk = self.full_vector(vkr)
            dfdrho += self.mass_matrix_derivative(rho, self.Q[:, k], vk)

            # Form the matrix
            def matrix(x):
                y = Ar.dot(x) - self.eigs[k] * Br.dot(x)
                t = np.dot(Ur.T, y)
                y = y - np.dot(Ur, t)
                return y

            matop = linalg.LinearOperator((nr, nr), matrix)

            # Solve the augmented system of equations for wk
            t = np.dot(Ur.T, bkr)
            bkr = bkr - np.dot(Ur, t)
            wkr, info = linalg.gmres(matop, bkr, M=preop, atol=1e-15, tol=1e-10)
            wk = self.full_vector(wkr)

            # Compute the contributions from the derivative from Adot
            dfdrho += 2.0 * self.stiffness_matrix_derivative(rho, self.Q[:, k], wk)

            # Add the contributions to the derivative from Bdot here...
            dfdrho -= self.eigs[k] * self.mass_matrix_derivative(rho, self.Q[:, k], wk)

            # Now, compute the remaining contributions to the derivative from
            # B by solving the second auxiliary system
            dkr = Ar.dot(vkr)

            t = np.dot(Ur.T, dkr)
            dkr = dkr - np.dot(Ur, t)
            ukr, info = linalg.gmres(matop, dkr, M=preop, atol=1e-15, tol=1e-10)
            uk = self.full_vector(ukr)

            # Compute the contributions from the derivative
            dfdrho -= self.mass_matrix_derivative(rho, self.Q[:, k], uk)

        return self.fltr.applyGradient(dfdrho, x)

    @time_this
    def get_stress_values_deriv(self, rho, eta_stress, eta, Q, allowable=1.0):

        dfdrhoE = np.zeros(self.nelems)

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        # Compute the stress relaxation factor
        relax_deriv = self.epsilon / (rhoE + self.epsilon * (1.0 - rhoE)) ** 2

        if self.assume_same_element:
            Be_ = np.zeros((3, 8))

            # Compute the x and y coordinates of each element
            xe_ = self.X[self.conn[0], 0]
            ye_ = self.X[self.conn[0], 1]

            # Compute the stress in the middle of the element
            xi = 0.0
            eta_ = 0.0
            _populate_Be_single(xi, eta_, xe_, ye_, Be_)

            for k in range(len(eta)):
                qe = np.zeros((self.nelems, 8))
                qe[:, ::2] = Q[2 * self.conn, k]
                qe[:, 1::2] = Q[2 * self.conn + 1, k]

                # Compute the stresses in each element
                s = np.einsum("ij,jk,nk -> ni", self.C0, Be_, qe)

                # Add the contributions from the von Mises stress
                dfdrhoE += (
                    eta[k]
                    * relax_deriv
                    * eta_stress
                    * (
                        s[:, 0] ** 2
                        + s[:, 1] ** 2
                        - s[:, 0] * s[:, 1]
                        + 3.0 * s[:, 2] ** 2
                    )
                ) / allowable**2

        else:
            Be = np.zeros((self.nelems, 3, 8))

            # Compute the x and y coordinates of each element
            xe = self.X[self.conn, 0]
            ye = self.X[self.conn, 1]

            # Compute the stress in the middle of the element
            xi = 0.0
            eta_ = 0.0
            _populate_Be(self.nelems, xi, eta_, xe, ye, Be)

            for k in range(len(eta)):
                qe = np.zeros((self.nelems, 8))
                qe[:, ::2] = Q[2 * self.conn, k]
                qe[:, 1::2] = Q[2 * self.conn + 1, k]

                # Compute the stresses in each element
                s = np.einsum("ij,njk,nk -> ni", self.C0, Be, qe)

                # Add the contributions from the von Mises stress
                dfdrhoE += (
                    eta[k]
                    * relax_deriv
                    * eta_stress
                    * (
                        s[:, 0] ** 2
                        + s[:, 1] ** 2
                        - s[:, 0] * s[:, 1]
                        + 3.0 * s[:, 2] ** 2
                    )
                ) / allowable**2

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return dfdrho

    @time_this
    def get_stress_product(self, rho, eta_stress, q, allowable=1.0):

        # Loop over all the eigenvalues
        # Dq = np.zeros(self.nvars)

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        # Compute the stress relaxation factor
        relax = rhoE / (rhoE + self.epsilon * (1.0 - rhoE))

        if self.assume_same_element:
            Be_ = np.zeros((3, 8))

            # Compute the x and y coordinates of each element
            xe_ = self.X[self.conn[0], 0]
            ye_ = self.X[self.conn[0], 1]

            # Compute the stress in the middle of the element
            xi = 0.0
            eta_ = 0.0
            _populate_Be_single(xi, eta_, xe_, ye_, Be_)

            qe = np.zeros((self.nelems, 8))
            qe[:, ::2] = q[2 * self.conn]
            qe[:, 1::2] = q[2 * self.conn + 1]

            # Compute the stresses in each element
            s = np.einsum("ij,jk,nk -> ni", self.C0, Be_, qe)

            ds = np.zeros((self.nelems, 3))
            ds[:, 0] = eta_stress * relax * (s[:, 0] - 0.5 * s[:, 1]) / allowable**2
            ds[:, 1] = eta_stress * relax * (s[:, 1] - 0.5 * s[:, 0]) / allowable**2
            ds[:, 2] = eta_stress * relax * 3.0 * s[:, 2] / allowable**2

            Dqe = np.einsum("ni,ij,jk -> nk", ds, self.C0, Be_)

        else:
            Be = np.zeros((self.nelems, 3, 8))

            # Compute the x and y coordinates of each element
            xe = self.X[self.conn, 0]
            ye = self.X[self.conn, 1]

            # Compute the stress in the middle of the element
            xi = 0.0
            eta_ = 0.0
            _populate_Be(self.nelems, xi, eta_, xe, ye, Be)

            qe = np.zeros((self.nelems, 8))
            qe[:, ::2] = q[2 * self.conn]
            qe[:, 1::2] = q[2 * self.conn + 1]

            # Compute the stresses in each element
            s = np.einsum("ij,njk,nk -> ni", self.C0, Be, qe)

            ds = np.zeros((self.nelems, 3))
            ds[:, 0] = eta_stress * relax * (s[:, 0] - 0.5 * s[:, 1]) / allowable**2
            ds[:, 1] = eta_stress * relax * (s[:, 1] - 0.5 * s[:, 0]) / allowable**2
            ds[:, 2] = eta_stress * relax * 3.0 * s[:, 2] / allowable**2

            Dqe = np.einsum("ni,ij,njk -> nk", ds, self.C0, Be)

        Dq = np.zeros(self.nvars)
        for i in range(4):
            np.add.at(Dq, 2 * self.conn[:, i], Dqe[:, 2 * i])
            np.add.at(Dq, 2 * self.conn[:, i] + 1, Dqe[:, 2 * i + 1])

        return Dq

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


class TopOptProb:
    """
    natural frequency maximization under a volume constraint
    """

    def __init__(
        self,
        analysis: TopologyAnalysis,
        non_design_nodes: list,  # indices for nodes whose density is not controlled by the optimizer
        vol_frac=0.4,
        ks_rho=100,
        m0=10.0,
        draw_history=True,
        draw_every=1,
        prefix="result",
        dv_mapping=None,  # If provided, optimizer controls reduced design variable xr only
        lb=1e-3,
        objf="frequency",
        confs="volume",
        omega_lb=None,
    ):
        self.analysis = analysis
        self.non_design_nodes = non_design_nodes
        self.xfull = np.zeros(self.analysis.nnodes)
        self.xfull[self.non_design_nodes] = 1.0
        self.design_nodes = np.ones(len(self.xfull), dtype=bool)
        self.design_nodes[self.non_design_nodes] = False
        self.dv_mapping = dv_mapping
        self.lb = lb
        self.objf = objf
        self.confs = confs
        self.omega_lb = omega_lb

        # Add more non-design constant to matrices
        self.add_mat0("M", non_design_nodes, density=m0)

        x = np.ones(self.analysis.nnodes)
        self.area_gradient = self.analysis.eval_area_gradient(x)
        self.fixed_area = vol_frac * np.sum(self.area_gradient)
        self.ks_rho = ks_rho

        self.ndv = np.sum(self.design_nodes)
        if dv_mapping is not None:
            self.ndv = dv_mapping.shape[1]

        self.ncon = 1
        if isinstance(confs, list):
            self.ncon = len(confs)

        self.draw_history = draw_history
        self.draw_every = draw_every
        self.prefix = prefix

        self.it_counter = 0
        return

    def add_mat0(self, which, non_design_nodes, density=1.0):
        assert which == "K" or which == "M"

        rho = np.zeros(self.analysis.nnodes)
        rho[non_design_nodes] = 1.0
        if which == "M":
            M0 = density * self.analysis.assemble_mass_matrix(rho)
            self.analysis.set_M0(M0)
            return

        # Else if which == "K":
        K0 = self.analysis.assemble_stiffness_matrix(rho)
        self.analysis.set_K0(K0)
        return

    def getVarsAndBounds(self, x, lb, ub):
        lb[:] = self.lb
        ub[:] = 1.0
        x[:] = 0.95
        # np.random.seed(0)
        # x[:] = 0.5 + 0.5 * np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x):
        t_start = timer()

        vtk_nodal_sols = None
        vtk_cell_sols = None
        vtk_path = None

        # Save the design to vtk every certain iterations
        if self.it_counter % self.draw_every == 0:
            if not os.path.isdir(os.path.join(self.prefix, "vtk")):
                os.mkdir(os.path.join(self.prefix, "vtk"))
            vtk_nodal_sols = {}
            vtk_cell_sols = {}
            vtk_path = os.path.join(self.prefix, "vtk", "it_%d.vtk" % self.it_counter)

        # Populate the nodal variable for analysis
        if self.dv_mapping is not None:
            self.xfull[:] = self.dv_mapping.dot(x)  # x = E*xr
            self.xfull[self.non_design_nodes] = 1.0
        else:
            self.xfull[self.design_nodes] = x[:]

        # Solve the genrealized eigenvalue problem
        omega = self.analysis.solve_eigenvalue_problem(
            self.xfull, k=6, nodal_sols=vtk_nodal_sols
        )

        # obj = self.analysis.eigenvector_displacement()

        if self.objf == "frequency":
            obj = -self.analysis.ks_omega(ks_rho=self.ks_rho)
        else:  # objf == "stress"
            obj = self.analysis.eigenvector_stress(self.xfull, cell_sols=vtk_cell_sols)

        # Compute constraints
        con = []
        if "volume" in self.confs:
            con.append(self.fixed_area - self.analysis.eval_area(self.xfull))

        if "frequency" in self.confs:
            assert self.omega_lb is not None
            omega_ks = self.analysis.ks_omega(ks_rho=self.ks_rho)
            con.append(omega_ks - self.omega_lb)

        # Save the design png and vtk
        if self.draw_history and self.it_counter % self.draw_every == 0:
            fig, ax = plt.subplots()
            rho = self.analysis.fltr.apply(self.xfull)
            self.analysis.plot(rho, ax=ax)
            ax.set_aspect("equal", "box")
            fig.savefig(os.path.join(self.prefix, "%d.png" % self.it_counter))
            plt.close()
            to_vtk(
                vtk_path,
                self.analysis.conn,
                self.analysis.X,
                vtk_nodal_sols,
                vtk_cell_sols,
            )

        # Log eigenvalues
        with open(os.path.join(self.prefix, "frequencies.log"), "a") as f:
            if self.it_counter % 10 == 0:
                f.write("\n%10s" % "iter")
                for i in range(len(omega)):
                    name = "omega[%d]" % i
                    f.write("%25s" % name)
                f.write("\n")

            f.write("%10d" % self.it_counter)
            for i in range(len(omega)):
                f.write("%25.15e" % omega[i])
            f.write("\n")

        t_end = timer()
        elapse_s = t_end - t_start
        print("[%3d], t(fun): %6.3f s, " % (self.it_counter, elapse_s), end="")

        fail = 0
        self.it_counter += 1
        return fail, obj, con

    def evalObjConGradient(self, x, g, A):
        t_start = timer()

        if self.dv_mapping is not None:
            # Populate the nodal variable for analysis
            self.xfull[:] = self.dv_mapping.dot(x)  # x = E*xr
            self.xfull[self.non_design_nodes] = 1.0

            if self.objf == "frequency":
                g[:] = -self.dv_mapping.T.dot(
                    self.analysis.ks_omega_derivative(self.xfull)
                )
            else:  # objf == "stress"
                g[:] = self.dv_mapping.T.dot(
                    self.analysis.eigenvector_stress_derivative(self.xfull)
                )

            # g[:] = self.dv_mapping.T.dot(
            #     self.analysis.eigenvector_displacement_deriv(self.xfull)
            # )

            # Evaluate constraint gradients
            index = 0
            if "volume" in args.confs:
                A[index][:] = -self.dv_mapping.T.dot(
                    self.analysis.eval_area_gradient(self.xfull)
                )
                index += 1

            if "frequency" in self.confs:
                A[index][:] = self.dv_mapping.T.dot(
                    self.analysis.ks_omega_derivative(self.xfull)
                )
                index += 1

        else:
            # Populate the nodal variable for analysis
            self.xfull[self.design_nodes] = x[:]

            # g[:] = self.analysis.eigenvector_displacement_deriv(self.xfull)[
            #     self.design_nodes
            # ]

            if self.objf == "frequency":
                g[:] = -self.analysis.ks_omega_derivative(self.xfull)[self.design_nodes]
            else:  # objf == "stress"
                g[:] = self.analysis.eigenvector_stress_derivative(self.xfull)[
                    self.design_nodes
                ]

            # Evaluate constraint gradient
            index = 0
            if "volume" in args.confs:
                A[index][:] = -self.analysis.eval_area_gradient(self.xfull)[
                    self.design_nodes
                ]
                index += 1

            if "frequency" in self.confs:
                A[index][:] = self.analysis.ks_omega_derivative(self.xfull)[
                    self.design_nodes
                ]
                index += 1

        t_end = timer()
        elapse_s = t_end - t_start
        print("t(grad): %6.3f s" % (elapse_s))

        return 0


try:
    import mma4py

    class MMAProblem(mma4py.Problem):
        def __init__(self, prob: TopOptProb) -> None:
            self.prob = prob
            self.ncon = prob.ncon
            super().__init__(prob.ndv, prob.ndv, prob.ncon)
            return

        def getVarsAndBounds(self, x, lb, ub):
            self.prob.getVarsAndBounds(x, lb, ub)
            return

        def evalObjCon(self, x, cons) -> float:
            _fail, _obj, _cons = self.prob.evalObjCon(x)
            for i in range(self.ncon):
                cons[i] = -_cons[i]
            return _obj

        def evalObjConGrad(self, x, g, gcon):
            self.prob.evalObjConGradient(x, g, gcon)
            for i in range(self.ncon):
                gcon[i, :] = -gcon[i, :]
            return

except:
    MMAProblem = None


class ParOptProblem(ParOpt.Problem):
    def __init__(self, comm, prob: TopOptProb) -> None:
        self.prob = prob
        super().__init__(comm, prob.ndv, prob.ncon)
        return

    def getVarsAndBounds(self, x, lb, ub):
        return self.prob.getVarsAndBounds(x, lb, ub)

    def evalObjCon(self, x):
        return self.prob.evalObjCon(x)

    def evalObjConGradient(self, x, g, A):
        return self.prob.evalObjConGradient(x, g, A)


def to_vtk(vtk_path, conn, X, nodal_sols={}, cell_sols={}):
    """
    Generate a vtk given conn, X, and optionally list of nodal solutions
    """
    # vtk requires a 3-dimensional data point
    X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)

    nnodes = X.shape[0]
    nelems = conn.shape[0]

    # Create a empty vtk file and write headers
    with open(vtk_path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("my example\n")
        fh.write("ASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")

        # Write nodal points
        fh.write("POINTS {:d} double\n".format(nnodes))
        for x in X:
            row = f"{x}"[1:-1]  # Remove square brackets in the string
            fh.write(f"{row}\n")

        # Write connectivity
        size = 5 * nelems

        fh.write(f"CELLS {nelems} {size}\n")
        for c in conn:
            node_idx = f"{c}"[1:-1]  # remove square bracket [ and ]
            npts = 4
            fh.write(f"{npts} {node_idx}\n")

        # Write cell type
        fh.write(f"CELL_TYPES {nelems}\n")
        for c in conn:
            vtk_type = 9
            fh.write(f"{vtk_type}\n")

        # Write solution
        if nodal_sols:
            fh.write(f"POINT_DATA {nnodes}\n")
            for name, data in nodal_sols.items():
                fh.write(f"SCALARS {name} float 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")

        if cell_sols:
            fh.write(f"CELL_DATA {nelems}\n")
            for name, data in cell_sols.items():
                fh.write(f"SCALARS {name} float 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")

    return


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


def create_square_domain(r0_, l=1.0, nx=30):
    """
    Args:
        l: length of the square
        nx: number of elements along x direction
    """

    # Generate the square domain problem by default
    m = nx
    n = nx

    nelems = m * n
    nnodes = (m + 1) * (n + 1)

    y = np.linspace(0, l, n + 1)
    x = np.linspace(0, l, m + 1)
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

    # We would like the center node or element to be the non-design region
    non_design_nodes = []
    for j in range(n // 2, (n + 1) // 2 + 1):
        for i in range(n // 2, (n + 1) // 2 + 1):
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

    r0 = l / nx * r0_

    # Create the mapping E such that x = E*xr, where xr is the nodal variable
    # of a quarter and is controlled by the optimizer, x is the nodal variable
    # of the entire domain
    Ei = []
    Ej = []
    redu_idx = 0

    # 8-way reflection
    for j in range(1, (n + 1) // 2):
        for i in range(j):
            if nodes[j, i] not in non_design_nodes:
                Ej.extend(8 * [redu_idx])
                Ei.extend(
                    [nodes[j, i], nodes[j, m - i], nodes[n - j, i], nodes[n - j, m - i]]
                )
                Ei.extend(
                    [nodes[i, j], nodes[i, m - j], nodes[n - i, j], nodes[n - i, m - j]]
                )
                redu_idx += 1

    # 4-way reflection of diagonals
    for i in range((n + 1) // 2):
        if nodes[i, i] not in non_design_nodes:
            Ej.extend(4 * [redu_idx])
            Ei.extend(
                [nodes[i, i], nodes[i, m - i], nodes[n - i, i], nodes[n - i, m - i]]
            )
            redu_idx += 1

    # 4-way reflection of x- and y-symmetry axes, only apply if number of elements
    # along x (and y) is even
    if n % 2 == 0:
        j = n // 2
        for i in range(j + 1):
            if nodes[i, j] not in non_design_nodes:
                Ej.extend(4 * [redu_idx])
                Ei.extend([nodes[i, j], nodes[n - i, j], nodes[j, i], nodes[j, n - i]])
                redu_idx += 1

    Ev = np.ones(len(Ei))
    dv_mapping = coo_matrix((Ev, (Ei, Ej)))

    return conn, X, r0, bcs, forces, non_design_nodes, dv_mapping


def visualize_domain(prefix, X, bcs, non_design_nodes=None):
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    bc_X = np.array([X[i, :] for i in bcs.keys()])
    ax.scatter(X[:, 0], X[:, 1], color="black")
    ax.scatter(bc_X[:, 0], bc_X[:, 1], color="red")

    if non_design_nodes:
        m0_X = np.array([X[i, :] for i in non_design_nodes])
        ax.scatter(m0_X[:, 0], m0_X[:, 1], color="blue")
    fig.savefig(os.path.join(prefix, "domain.pdf"))
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


def parse_cmd_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # OS
    p.add_argument("--prefix", default="result", type=str, help="result folder")

    # Analysis
    p.add_argument(
        "--assume-same-element",
        action="store_true",
        help="assume all elements have identical shape",
    )
    p.add_argument(
        "--nx", default=96, type=int, help="number of elements along x direction"
    )
    p.add_argument(
        "--lb", default=1e-3, type=float, help="lower bound of the design variable"
    )
    p.add_argument(
        "--filter",
        default="spatial",
        choices=["spatial", "helmholtz"],
        help="density filter type",
    )
    p.add_argument("--ks-rho", default=10000, type=int, help="ks aggregation parameter")
    p.add_argument(
        "--ptype",
        default="ramp",
        choices=["simp", "ramp"],
        help="material penalization method",
    )
    p.add_argument(
        "--p", default=5.0, type=float, help="material penalization parameter"
    )
    p.add_argument(
        "--m0", default=100.0, type=float, help="magnitude of non-design mass"
    )
    p.add_argument("--r0", default=2.1, type=float, help="filter radius = r0 * lx / nx")

    # Optimization
    p.add_argument(
        "--optimizer",
        default="mma4py",
        choices=["pmma", "mma4py", "tr"],
        help="optimization method",
    )
    p.add_argument(
        "--objf",
        default="frequency",
        choices=["frequency", "stress"],
        help="objective function",
    )
    p.add_argument(
        "--confs",
        default="volume",
        nargs="*",
        choices=["volume", "frequency"],
        help="constraint functions",
    )
    p.add_argument(
        "--omega-lb",
        default=None,
        type=float,
        help='Lower bound of natural frequency, only effective when "frequency" is in the confs',
    )
    p.add_argument(
        "--maxit", default=200, type=int, help="maximum number of iterations"
    )
    p.add_argument(
        "--grad-check", action="store_true", help="perform gradient check and exit"
    )
    args = p.parse_args()

    return args


if __name__ == "__main__":
    # Get options from command line
    args = parse_cmd_args()

    # Create result directory if needed
    if not os.path.isdir(args.prefix):
        os.mkdir(args.prefix)

    # Save option values
    with open(os.path.join(args.prefix, "options.txt"), "w") as f:
        f.write("Options:\n")
        for k, v in vars(args).items():
            f.write(f"{k:<20}{v}\n")

    conn, X, r0, bcs, forces, non_design_nodes, dv_mapping = create_square_domain(
        r0_=args.r0, nx=args.nx
    )

    # Check the mesh
    visualize_domain(args.prefix, X, bcs, non_design_nodes)

    # Create the filter
    fltr = NodeFilter(conn, X, r0, ftype=args.filter, projection=False)

    # Create analysis
    analysis = TopologyAnalysis(
        fltr,
        conn,
        X,
        bcs,
        forces,
        ptype=args.ptype,
        p=args.p,
        assume_same_element=args.assume_same_element,
    )

    # Create optimization problem
    topo = TopOptProb(
        analysis,
        non_design_nodes,
        ks_rho=args.ks_rho,
        m0=args.m0,
        draw_every=5,
        prefix=args.prefix,
        dv_mapping=dv_mapping,
        lb=args.lb,
        objf=args.objf,
        confs=args.confs,
        omega_lb=args.omega_lb,
    )

    # Print info
    print("=== Problem overview ===")
    print("objective:   %s" % args.objf)
    print(f"constraints: {args.confs}")
    print("num of dof:  %d" % analysis.nvars)
    print("num of dv:   %d" % topo.ndv)
    print()

    if args.optimizer == "mma4py":

        if MMAProblem is None:
            raise ImportError("Cannot use mma4py, package not found.")

        mmaprob = MMAProblem(topo)
        mmaopt = mma4py.Optimizer(
            mmaprob, log_name=os.path.join(args.prefix, "mma4py.log")
        )
        mmaopt.checkGradients()
        if args.grad_check:
            exit(0)

        mmaopt.optimize(niter=args.maxit, verbose=False)

    else:
        from mpi4py import MPI

        paroptprob = ParOptProblem(MPI.COMM_SELF, topo)
        paroptprob.checkGradients(1e-6)
        if args.grad_check:
            exit(0)

        if args.optimizer == "pmma":
            algorithm = "mma"
        else:
            algorithm = args.optimizer
        options = get_paropt_default_options(
            args.prefix, algorithm=algorithm, maxit=args.maxit
        )

        # Set up the optimizer
        opt = ParOpt.Optimizer(paroptprob, options)

        # Set a new starting point
        opt.optimize()
        x, z, zw, zl, zu = opt.getOptimizedPoint()
