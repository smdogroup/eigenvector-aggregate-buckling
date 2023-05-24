import argparse
from collections import OrderedDict
import multiprocessing as multiprocessing
import os
import shutil as shutil
from shutil import rmtree
from timeit import default_timer as timer

from icecream import ic
import matplotlib.pylab as plt
import matplotlib.tri as tri
import mpmath as mp
import numpy as np
from paropt import ParOpt
from scipy import sparse, spatial
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, linalg

from other.utils import time_this, timer_set_log_path


class Logger:
    log_name = "stdout.log"

    @staticmethod
    def set_log_path(log_path):
        Logger.log_name = log_path

    @staticmethod
    def log(txt="", end="\n"):
        with open(Logger.log_name, "a") as f:
            f.write("%s%s" % (txt, end))
        return


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


def _populate_Be_and_Te(nelems, xi, eta, xe, ye, Be, Te):
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

    # Set the entries for the stress stiffening matrix
    for i in range(nelems):
        Te[i, 0, :, :] = np.outer(Nx[i, :], Nx[i, :])
        Te[i, 1, :, :] = np.outer(Ny[i, :], Ny[i, :])
        Te[i, 2, :, :] = np.outer(Nx[i, :], Ny[i, :]) + np.outer(Ny[i, :], Nx[i, :])

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


def _populate_Be_and_Te_single(xi, eta, xe, ye, Be, Te):
    """
    Populate B matrices for all elements at a quadrature point
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
    invJ[0, 0] = J[1, 1] / detJ
    invJ[0, 1] = -J[0, 1] / detJ
    invJ[1, 0] = -J[1, 0] / detJ
    invJ[1, 1] = J[0, 0] / detJ

    # Compute the derivative of the shape functions w.r.t. xi and eta
    # [Nx, Ny] = [Nxi, Neta]*invJ
    Nx = np.outer(invJ[0, 0], Nxi) + np.outer(invJ[1, 0], Neta)
    Ny = np.outer(invJ[0, 1], Nxi) + np.outer(invJ[1, 1], Neta)

    # Set the B matrix for each element
    Be[0, ::2] = Nx
    Be[1, 1::2] = Ny
    Be[2, ::2] = Ny
    Be[2, 1::2] = Nx

    # Set the entries for the stress stiffening matrix
    Te[0, :, :] = np.outer(Nx, Nx)
    Te[1, :, :] = np.outer(Ny, Ny)
    Te[2, :, :] = np.outer(Nx, Ny) + np.outer(Ny, Nx)

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
        m=None,
        D_index=None,
        E=10.0 * 1e6,
        nu=0.3,
        ptype_K="ramp",
        ptype_M="ramp",
        rho0_K=1e-3,
        rho0_M=1e-7,
        p=3.0,
        q=5.0,
        density=1.0,
        epsilon=0.3,
        assume_same_element=False,
    ):
        self.ptype_K = ptype_K.lower()
        self.ptype_M = ptype_M.lower()

        self.rho0_K = rho0_K
        self.rho0_M = rho0_M

        self.fltr = fltr
        self.conn = np.array(conn)
        self.X = np.array(X)
        self.p = p
        self.q = q
        self.density = density
        self.epsilon = epsilon
        self.assume_same_element = assume_same_element

        # C1 continuous mass penalization coefficients if ptype_M == msimp
        self.simp_c1 = 6e5
        self.simp_c2 = -5e6

        self.nelems = self.conn.shape[0]
        self.nnodes = int(np.max(self.conn)) + 1
        self.nvars = 2 * self.nnodes

        self.D_index = D_index
        self.K0 = None
        self.M0 = None

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
        if self.ptype_K == "simp":
            C = np.outer(rhoE**self.p + self.rho0_K, self.C0)
        else:  # ramp
            C = np.outer(rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0)

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

        if self.ptype_K == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1.0)
        else:  # ramp
            dfdrhoE[:] *= (1.0 + self.q) / (1.0 + self.q * (1.0 - rhoE)) ** 2

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
        if self.ptype_M == "msimp":
            nonlin = self.simp_c1 * rhoE**6.0 + self.simp_c2 * rhoE**7.0
            cond = (rhoE > 0.1).astype(int)
            density = self.density * (rhoE * cond + nonlin * (1 - cond))
        elif self.ptype_M == "ramp":
            density = self.density * (
                (self.q + 1.0) * rhoE / (1 + self.q * rhoE) + self.rho0_M
            )
        else:  # linear
            density = self.density * rhoE

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

        if self.ptype_M == "msimp":
            dnonlin = (
                6.0 * self.simp_c1 * rhoE**5.0 + 7.0 * self.simp_c2 * rhoE**6.0
            )
            cond = (rhoE > 0.1).astype(int)
            dfdrhoE[:] *= self.density * (cond + dnonlin * (1 - cond))
        elif self.ptype_M == "ramp":
            dfdrhoE[:] *= self.density * (1.0 + self.q) / (1.0 + self.q * rhoE) ** 2
        else:  # linear
            dfdrhoE[:] *= self.density

        dfdrho = np.zeros(self.nnodes)
        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        return dfdrho

    def assemble_stress_stiffness(self, rho, u):
        """
        Compute the stress stiffness matrix for buckling, given the displacement path u
        """

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        # Get the element-wise solution variables
        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            C = np.outer(rhoE ** (self.p + 1) + self.rho0_K, self.C0)
        else:  # ramp
            C = np.outer(
                rhoE / (1.0 + (self.q + 1) * (1.0 - rhoE)) + self.rho0_K, self.C0
            )

        C = C.reshape((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # Assemble all of the the 8 x 8 element stiffness matrix
        Ge = np.zeros((self.nelems, 8, 8))

        if self.assume_same_element:
            Be_ = np.zeros((3, 8))
            Te_ = np.zeros((3, 4, 4))

            # Compute the x and y coordinates of the first element
            xe_ = self.X[self.conn[0], 0]
            ye_ = self.X[self.conn[0], 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_Be_and_Te_single(xi, eta, xe_, ye_, Be_, Te_)

                    # Compute the stress
                    s = np.einsum("nij,jk,nk - > ni", C, Be_, ue)

                    # G0e =
                    # s[0] * np.outer(Nx, Nx) +
                    # s[1] * np.outer(Ny, Ny) +
                    # s[2] * (np.outer(Nx, Ny) + np.outer(Ny, Nx))
                    G0e = np.einsum("n,ni,ijl -> njl", detJ, s, Te_)
                    Ge[:, 0::2, 0::2] += G0e
                    Ge[:, 1::2, 1::2] += G0e
        else:
            Be = np.zeros((self.nelems, 3, 8))
            Te = np.zeros((self.nelems, 3, 4, 4))

            # Compute the x and y coordinates of each element
            xe = self.X[self.conn, 0]
            ye = self.X[self.conn, 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_Be_and_Te(self.nelems, xi, eta, xe, ye, Be, Te)

                    # Compute the stresses in each element
                    s = np.einsum("nij,njk,nk -> ni", C, Be, ue)

                    # G0e =
                    # s[0] * np.outer(Nx, Nx) +
                    # s[1] * np.outer(Ny, Ny) +
                    # s[2] * (np.outer(Nx, Ny) + np.outer(Ny, Nx))
                    G0e = np.einsum("n,ni,nijl -> njl", detJ, s, Te)
                    Ge[:, 0::2, 0::2] += G0e
                    Ge[:, 1::2, 1::2] += G0e

        # Assemble the global stiffness matrix
        G = sparse.coo_matrix((Ge.flatten(), (self.i, self.j)))
        G = G.tocsr()

        return G

    def stress_stiffness_derivative(self, rho, u, psi, phi, solver):
        """
        Compute the derivative of psi^{T} * G(u, x) * phi using the adjoint method.

        Note "solver" returns the solution of the system of equations

        K * sol = rhs

        Given the right-hand-side rhs. ie. sol = solver(rhs)
        """

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        dfdrho = np.zeros(self.nnodes)
        dfdC = np.zeros((self.nelems, 3, 3))

        # Compute the element stiffnesses
        if self.ptype_K == "simp":
            C = np.outer(rhoE ** (self.p + 1) + self.rho0_K, self.C0)
        else:  # ramp
            C = np.outer(
                rhoE / (1.0 + (self.q + 1) * (1.0 - rhoE)) + self.rho0_K, self.C0
            )

        C = C.reshape((self.nelems, 3, 3))

        # Compute the element stiffness matrix
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

        # The element-wise variables
        ue = np.zeros((self.nelems, 8))
        ue[:, ::2] = u[2 * self.conn]
        ue[:, 1::2] = u[2 * self.conn + 1]

        # Compute the element-wise values of psi and phi
        psie = np.zeros((self.nelems, 8))
        psie[:, ::2] = psi[2 * self.conn]
        psie[:, 1::2] = psi[2 * self.conn + 1]

        phie = np.zeros((self.nelems, 8))
        phie[:, ::2] = psi[2 * self.conn]
        phie[:, 1::2] = psi[2 * self.conn + 1]

        dfdue = np.zeros((self.nelems, 8))

        if self.assume_same_element:
            pass
        else:
            # Compute
            Be = np.zeros((self.nelems, 3, 8))
            Te = np.zeros((self.nelems, 3, 4, 4))

            # Compute the x and y coordinates of each element
            xe = self.X[self.conn, 0]
            ye = self.X[self.conn, 1]

            for j in range(2):
                for i in range(2):
                    xi = gauss_pts[i]
                    eta = gauss_pts[j]
                    detJ = _populate_Be_and_Te(self.nelems, xi, eta, xe, ye, Be, Te)

                    # Compute the stresses in each element
                    s = np.einsum("nij,njk,nk -> ni", C, Be, ue)

                    # Compute the derivative of the stress w.r.t. u
                    dfds = np.einsum(
                        "n,nijl,nj,nl -> ni", detJ, Te, psie[:, ::2], phie[:, ::2]
                    ) + np.einsum(
                        "n,nijl,nj,nl -> ni", detJ, Te, psie[:, 1::2], phie[:, 1::2]
                    )

                    # Add up contributions to d( psi^{T} * G(x, u) * phi ) / du
                    dfdue += np.einsum("nij,nik,nk -> nj", Be, C, dfds)

                    # Add contributions to the derivative w.r.t. C
                    dfdC += np.einsum("ni,njk,nk -> nij", dfds, Be, ue)

        dfdu = np.zeros(2 * self.nnodes)
        np.add.at(dfdu, 2 * self.conn, dfdue[:, 0::2])
        np.add.at(dfdu, 2 * self.conn + 1, dfdue[:, 1::2])
        dfdur = self.reduce_vector(dfdu)

        # Compute the adjoint for K * adj = d ( psi^{T} * G(u, x) * phi ) / du
        adjr = solver(dfdur)
        adj = self.full_vector(adjr)

        dfdrhoE = np.zeros(self.nelems)
        for i in range(3):
            for j in range(3):
                dfdrhoE[:] += self.C0[i, j] * dfdC[:, i, j]

        if self.ptype_K == "simp":
            dfdrhoE[:] *= (self.p + 1) * rhoE**self.p
        else:  # ramp
            dfdrhoE[:] *= (2.0 + self.q) / (1.0 + (self.q + 1.0) * (1.0 - rhoE)) ** 2

        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        # Compute the derivative of the stiffness matrix w.r.t. rho
        dfdrho -= self.stiffness_matrix_derivative(rho, adj, u)
        
        return dfdrho
    
    def check_buckling(self, rho, psi, phi, dh=1e-6):
        K = self.assemble_stiffness_matrix(rho)
        Kr = self.reduce_matrix(K)

        # Compute the solution path
        fr = self.reduce_vector(self.f)
        Kfact = linalg.factorized(Kr)
        ur = Kfact(fr)
        u = self.full_vector(ur)
        
        # Find the gemoetric stiffness matrix
        G = self.assemble_stress_stiffness(rho, u)
        Gr = self.reduce_matrix(G)
        
        gx = self.stress_stiffness_derivative(rho, u, psi, phi, Kfact)
        
        f0 = np.dot(psi, G @ phi)
        p_rho = np.random.uniform(size=rho.shape)
        rho_1 = rho + dh * p_rho
        exact = np.dot(gx, p_rho)
        
        K = self.assemble_stiffness_matrix(rho_1)
        Kr = self.reduce_matrix(K)

        # Compute the solution path
        fr = self.reduce_vector(self.f)
        Kfact = linalg.factorized(Kr)
        ur = Kfact(fr)
        u = self.full_vector(ur)
        
        # Find the gemoetric stiffness matrix
        G = self.assemble_stress_stiffness(rho_1, u)
        Gr = self.reduce_matrix(G)
        
        f1 = np.dot(psi, G @ phi)
        fd = (f1 - f0) / dh
        
        print("Exact: ", exact)
        print("FD: ", fd)
        print("Error: ", np.abs(exact - fd)/np.abs(exact))
        

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
    def solve_eigenvalue_problem(
        self, x, k=5, sigma=0.0, nodal_sols=None, nodal_vecs=None
    ):
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

        if nodal_vecs is not None:
            for i in range(k):
                nodal_vecs["phi%d" % i] = [Q[0::2, i], Q[1::2, i]]

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

    def solve_buckling(self, x, k=5, sigma=1000, nodal_sols=None, nodal_vecs=None):
        if k > len(self.reduced):
            k = len(self.reduced)

        # Compute the density at each node
        rho = self.fltr.apply(x)

        K = self.assemble_stiffness_matrix(rho)
        Kr = self.reduce_matrix(K)

        # Compute the solution path
        fr = self.reduce_vector(self.f)
        Kfact = linalg.factorized(Kr)
        ur = Kfact(fr)
        u = self.full_vector(ur)

        # Find the gemoetric stiffness matrix
        G = self.assemble_stress_stiffness(rho, u)
        Gr = self.reduce_matrix(G)

        # Find the eigenvalues closest to zero. This uses a shift and
        # invert strategy around sigma = 0, which means that the largest
        # magnitude values are closest to zero.
        if k == len(self.reduced):
            # eigs, Qr = eigh(Kr.todense(), Gr.todense())
            pass
        else:
            eigs, Qr = sparse.linalg.eigsh(
                Kr, M=Gr, k=k, sigma=-sigma, mode="buckling", which="SM", tol=1e-10
            )
     
        eigs *= -1
        R = Kr @ Qr[:, 0] + eigs[0] * Gr @ Qr[:, 0]
        
        self.QGQ = np.zeros(k)
        for i in range(k):
            self.QGQ[i] = np.dot(Qr[:, i], Gr @ Qr[:, i])

        Q = np.zeros((self.nvars, k))
        for i in range(k):
            Q[self.reduced, i] = Qr[:, i]

        # Save vtk output data
        if nodal_sols is not None:
            nodal_sols["x"] = np.array(x)
            nodal_sols["rho"] = np.array(rho)

        if nodal_vecs is not None:
            for i in range(k):
                nodal_vecs["phi%d" % i] = [Q[0::2, i], Q[1::2, i]]

        # Save the eigenvalues and eigenvectors
        self.eigs = eigs
        self.Q = Q
        
        self.check_buckling(rho, Q[:,0], Q[:,0])

        return eigs

    def ks_buckling(self, ks_rho=100.0):
        """
        Compute the smoothed approximation of the smallest eigenvalue
        """
        c = np.min(self.eigs)
        eta = np.exp(-ks_rho * (self.eigs - c))
        a = np.sum(eta)
        ks_min = c - np.log(a) / ks_rho

        return ks_min

    def ks_buckling_derivative(self, x, ks_rho=100.0):
        # Compute the density at each node
        rho = self.fltr.apply(x)

        c = np.min(self.eigs)
        eta = np.exp(-ks_rho * (self.eigs - c))
        a = np.sum(eta)
        eta *= 1.0 / a

        # Solve for the load path again here...
        K = self.assemble_stiffness_matrix(rho)
        Kr = self.reduce_matrix(K)
        fr = self.reduce_vector(self.f)
        Kfact = linalg.factorized(Kr)
        ur = Kfact(fr)
        u = self.full_vector(ur)

        dfdrho = np.zeros(self.nnodes)
        Q = self.Q

        for i in range(len(self.eigs)):
            kx = self.stiffness_matrix_derivative(rho, Q[:, i], Q[:, i])
            gx = self.stress_stiffness_derivative(rho, u, Q[:, i], Q[:, i], Kfact)
            dfdrho -= eta[i] * (kx + self.eigs[i] * gx) / self.QGQ[i]

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
            q = self.Q[self.D_index, i]
            h += eta[i] * np.dot(q, q)

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
            q = self.Q[self.D_index, i]
            h += eta[i] * np.dot(q, q)

        # Set the value of the derivative
        dfdrho = np.zeros(self.nnodes)

        for j in range(N):
            for i in range(j + 1):
                qDq = np.dot(self.Q[self.D_index, i], self.Q[self.D_index, j])
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
            # Compute B * uk = D * qk
            bk = np.zeros(self.nvars)
            bk[self.D_index] = self.Q[self.D_index, k]
            bkr = -eta[k] * self.reduce_vector(bk)

            ukr = Bfact(bkr)
            uk = self.full_vector(ukr)
            dfdrho += self.mass_matrix_derivative(rho, self.Q[:, k], uk)

            # Form the matrix
            def matrix(x):
                y = Ar.dot(x) - self.eigs[k] * Br.dot(x)
                t = np.dot(Ur.T, y)
                y = y - np.dot(Ur, t)
                return y

            matop = linalg.LinearOperator((nr, nr), matrix)

            # Solve the augmented system of equations for vk
            t = np.dot(Ur.T, bkr)
            bkr = bkr - np.dot(Ur, t)
            vkr, info = linalg.gmres(matop, bkr, M=preop, atol=1e-15, tol=1e-10)
            vk = self.full_vector(vkr)

            # Compute the contributions from the derivative from Adot
            dfdrho += 2.0 * self.stiffness_matrix_derivative(rho, self.Q[:, k], vk)

            # Add the contributions to the derivative from Bdot here...
            dfdrho -= self.eigs[k] * self.mass_matrix_derivative(rho, self.Q[:, k], vk)

            # Now, compute the remaining contributions to the derivative from
            # B by solving the second auxiliary system
            dkr = Ar.dot(ukr)

            t = np.dot(Ur.T, dkr)
            dkr = dkr - np.dot(Ur, t)
            wkr, info = linalg.gmres(matop, dkr, M=preop, atol=1e-15, tol=1e-10)
            wk = self.full_vector(wkr)

            # Compute the contributions from the derivative
            dfdrho -= self.mass_matrix_derivative(rho, self.Q[:, k], wk)

        return self.fltr.applyGradient(dfdrho, x)

    @time_this
    def postprocess_strain_stress(self, x, u, allowable=1.0):
        """
        Compute strain field and Von-mises stress given a displacement field
        """
        # Compute the filtered variables
        rho = self.fltr.apply(x)

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            rho[self.conn[:, 0]]
            + rho[self.conn[:, 1]]
            + rho[self.conn[:, 2]]
            + rho[self.conn[:, 3]]
        )

        # Compute the stress relaxation factor
        relax = rhoE / (rhoE + self.epsilon * (1.0 - rhoE))

        Be = np.zeros((self.nelems, 3, 8))

        # Compute the x and y coordinates of each element
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        # Compute the stress in the middle of the element
        xi = 0.0
        eta = 0.0
        _populate_Be(self.nelems, xi, eta, xe, ye, Be)

        qe = np.zeros((self.nelems, 8))
        qe[:, ::2] = u[2 * self.conn]
        qe[:, 1::2] = u[2 * self.conn + 1]

        # Compute the stresses in each element
        strain = np.einsum("nik,nk -> ni", Be, qe)
        s = np.einsum("ij,njk,nk -> ni", self.C0, Be, qe)

        # Compute the von Mises stress
        vonmises = (
            relax
            * (s[:, 0] ** 2 + s[:, 1] ** 2 - s[:, 0] * s[:, 1] + 3.0 * s[:, 2] ** 2)
        ) / allowable**2

        return strain, vonmises

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
            cell_sols["eigenvector_stress"] = stress

        # Now aggregate over the stress
        max_stress = np.max(stress)
        h = max_stress + np.log(np.sum(np.exp(ks_rho * (stress - max_stress)))) / ks_rho

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
            # Compute B * uk = D * qk
            bk = self.get_stress_product(
                rho, eta_stress, self.Q[:, k], allowable=allowable
            )
            bkr = -eta[k] * self.reduce_vector(bk)

            ukr = Bfact(bkr)
            uk = self.full_vector(ukr)
            dfdrho += self.mass_matrix_derivative(rho, self.Q[:, k], uk)

            # Form the matrix
            def matrix(x):
                y = Ar.dot(x) - self.eigs[k] * Br.dot(x)
                t = np.dot(Ur.T, y)
                y = y - np.dot(Ur, t)
                return y

            matop = linalg.LinearOperator((nr, nr), matrix)

            # Solve the augmented system of equations for vk
            t = np.dot(Ur.T, bkr)
            bkr = bkr - np.dot(Ur, t)
            vkr, info = linalg.gmres(matop, bkr, M=preop, atol=1e-15, tol=1e-10)
            vk = self.full_vector(vkr)

            # Compute the contributions from the derivative from Adot
            dfdrho += 2.0 * self.stiffness_matrix_derivative(rho, self.Q[:, k], vk)

            # Add the contributions to the derivative from Bdot here...
            dfdrho -= self.eigs[k] * self.mass_matrix_derivative(rho, self.Q[:, k], vk)

            # Now, compute the remaining contributions to the derivative from
            # B by solving the second auxiliary system
            dkr = Ar.dot(ukr)

            t = np.dot(Ur.T, dkr)
            dkr = dkr - np.dot(Ur, t)
            wkr, info = linalg.gmres(matop, dkr, M=preop, atol=1e-15, tol=1e-10)
            wk = self.full_vector(wkr)

            # Compute the contributions from the derivative
            dfdrho -= self.mass_matrix_derivative(rho, self.Q[:, k], wk)

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
        vol_frac_ub=None,
        vol_frac_lb=None,
        dis_ub=None,
        ks_rho=100,
        m0=10.0,
        draw_history=True,
        draw_every=1,
        prefix="output",
        dv_mapping=None,  # If provided, optimizer controls reduced design variable xr only
        lb=1e-6,
        prob="natural_frequency",
        objf="frequency",
        confs="volume_ub",
        omega_lb=None,
        stress_ub=None,
        stress_scale=1.0,
        grad_check=False,
        domain="square",
    ):
        self.analysis = analysis
        self.non_design_nodes = non_design_nodes
        self.xfull = np.zeros(self.analysis.nnodes)
        self.xfull[self.non_design_nodes] = 1.0
        self.design_nodes = np.ones(len(self.xfull), dtype=bool)
        self.design_nodes[self.non_design_nodes] = False
        self.dv_mapping = dv_mapping
        self.prob = prob
        self.objf = objf
        self.confs = confs
        self.omega_lb = omega_lb
        self.stress_ub = stress_ub
        self.stress_scale = stress_scale
        self.lb = lb
        self.grad_check = grad_check
        self.domain = domain

        # Add more non-design constant to matrices
        self.add_mat0("M", non_design_nodes, density=m0)

        x = np.ones(self.analysis.nnodes)
        self.area_gradient = self.analysis.eval_area_gradient(x)
        if vol_frac_ub is None:
            self.fixed_area_ub = None
        else:
            self.fixed_area_ub = vol_frac_ub * np.sum(self.area_gradient)

        if vol_frac_lb is None:
            self.fixed_area_lb = None
        else:
            self.fixed_area_lb = vol_frac_lb * np.sum(self.area_gradient)

        self.fixed_dis_ub = dis_ub

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
        if self.grad_check:
            np.random.seed(0)
            x[:] = 0.5 + 0.5 * np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x, eval_all=False):
        # Functions of interest to be logged
        foi = OrderedDict()
        foi["area"] = "n/a"
        foi["omega_ks"] = "n/a"
        foi["stress_ks"] = "n/a"

        t_start = timer()

        vtk_nodal_sols = None
        vtk_nodal_vecs = None
        vtk_cell_sols = None
        vtk_cell_vecs = None
        vtk_path = None
        stress_ks = None

        # Save the design to vtk every certain iterations
        if eval_all or self.it_counter % self.draw_every == 0:
            if not os.path.isdir(os.path.join(self.prefix, "vtk")):
                os.mkdir(os.path.join(self.prefix, "vtk"))
            vtk_nodal_sols = {}
            vtk_nodal_vecs = {}
            vtk_cell_sols = {}
            vtk_cell_vecs = {}
            vtk_path = os.path.join(self.prefix, "vtk", "it_%d.vtk" % self.it_counter)

        # Populate the nodal variable for analysis
        if self.dv_mapping is not None:
            self.xfull[:] = self.dv_mapping.dot(x)  # x = E*xr
            self.xfull[self.non_design_nodes] = 1.0
        else:
            self.xfull[self.design_nodes] = x[:]

        # Solve the genrealized eigenvalue problem
        if self.prob == "natural_frequency":
            omega = self.analysis.solve_eigenvalue_problem(
                self.xfull, k=6, nodal_sols=vtk_nodal_sols, nodal_vecs=vtk_nodal_vecs
            )
        elif self.prob == "buckling":
            omega = self.analysis.solve_buckling(
                self.xfull, k=6, nodal_sols=vtk_nodal_sols, nodal_vecs=vtk_nodal_vecs
            )

        # Evaluate objectives
        if self.objf == "volume":
            area = self.analysis.eval_area(self.xfull)
            obj = area
            if self.domain == "beam" or self.domain == "building":
                area *= 1 / np.sum(self.area_gradient)
            foi["area"] = area

        elif self.objf == "frequency":
            if self.prob == "natural_frequency":
                omega_ks = self.analysis.ks_omega(ks_rho=self.ks_rho)
            elif self.prob == "buckling":
                omega_ks = self.analysis.ks_buckling(ks_rho=self.ks_rho)
            obj = -1 * omega_ks
            foi["omega_ks"] = omega_ks
        elif self.objf == "compliance":
            compliance = self.analysis.compliance(self.xfull)
            obj = compliance
            foi["compliance"] = compliance
        elif self.objf == "displacement":
            displacement = self.analysis.eigenvector_displacement(ks_rho=self.ks_rho)
            obj = 10 * displacement
            foi["displacement"] = displacement
        else:  # objf == "stress"
            stress_ks = self.analysis.eigenvector_stress(
                self.xfull, cell_sols=vtk_cell_sols
            )
            obj = stress_ks * self.stress_scale
            foi["stress_ks"] = stress_ks

        # Compute constraints
        con = []
        if "volume_ub" in self.confs:
            assert self.fixed_area_ub is not None
            area = self.analysis.eval_area(self.xfull)
            con.append(self.fixed_area_ub - area)
            foi["area"] = area / np.sum(self.area_gradient)
            # stress_ks = self.analysis.eigenvector_stress(
            #     self.xfull, cell_sols=vtk_cell_sols
            # )
            # foi["stress_ks"] = stress_ks

        if "volume_lb" in self.confs:
            assert self.fixed_area_lb is not None
            area = self.analysis.eval_area(self.xfull)
            con.append(area - self.fixed_area_lb)
            if "volume_ub" not in self.confs:
                foi["area"] = area / np.sum(self.area_gradient)

        if "frequency" in self.confs:
            assert self.omega_lb is not None
            if self.prob == "natural_frequency":
                omega_ks = self.analysis.ks_omega(ks_rho=self.ks_rho)
            elif self.prob == "buckling":
                omega_ks = self.analysis.ks_buckling(ks_rho=self.ks_rho)
            con.append(omega_ks - self.omega_lb)
            foi["omega_ks"] = omega_ks

        if "stress" in self.confs:
            assert self.stress_ub is not None
            stress_ks = self.analysis.eigenvector_stress(
                self.xfull, cell_sols=vtk_cell_sols
            )
            con.append(1.0 - stress_ks / self.stress_ub)
            foi["stress_ks"] = stress_ks

        if "displacement" in self.confs:
            assert self.fixed_dis_ub is not None
            displacement = self.analysis.eigenvector_displacement(ks_rho=self.ks_rho)
            con.append(self.fixed_dis_ub - displacement)
            foi["displacement"] = displacement

        # Evaluate all quantities of interest
        if eval_all:
            stress_ks = self.analysis.eigenvector_stress(
                self.xfull, cell_sols=vtk_cell_sols
            )
            if self.prob == "natural_frequency":
                omega_ks = self.analysis.ks_omega(ks_rho=self.ks_rho)
            elif self.prob == "buckling":
                omega_ks = self.analysis.ks_buckling(ks_rho=self.ks_rho)
            foi["stress_ks"] = stress_ks
            foi["omega_ks"] = omega_ks

        # Save the design png and vtk
        if eval_all or (self.draw_history and self.it_counter % self.draw_every == 0):
            fig, ax = plt.subplots(figsize=(4.8, 4.8), constrained_layout=True)
            ax.set_xticks([])
            ax.set_yticks([])
            rho = self.analysis.fltr.apply(self.xfull)
            self.analysis.plot(rho, ax=ax)
            ax.set_aspect("equal", "box")
            fig.savefig(
                os.path.join(self.prefix, "%d.png" % self.it_counter),
                bbox_inches="tight",
            )
            plt.close()

        # Save the design png and vtk
        if eval_all or (
            self.draw_history and self.it_counter % (4 * self.draw_every) == 0
        ):
            # Save strain and von mises stress for each eigenmode
            for i, eig_mode in enumerate(self.analysis.Q.T):
                strain, vonmises = self.analysis.postprocess_strain_stress(
                    self.xfull, eig_mode
                )
                vtk_cell_sols["exx_%d" % i] = strain[:, 0]
                vtk_cell_sols["eyy_%d" % i] = strain[:, 1]
                vtk_cell_sols["exy_%d" % i] = strain[:, 2]
                vtk_cell_sols["vonmises_%d" % i] = vonmises[:]
            to_vtk(
                vtk_path,
                self.analysis.conn,
                self.analysis.X,
                vtk_nodal_sols,
                vtk_cell_sols,
                vtk_nodal_vecs,
                vtk_cell_vecs,
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
        self.elapse_f = t_end - t_start

        # Log function values and time
        if self.it_counter % 10 == 0:
            Logger.log("\n%5s%20s" % ("iter", "obj"), end="")
            for k in foi.keys():
                Logger.log("%20s" % k, end="")
            Logger.log("%10s%10s" % ("tfun(s)", "tgrad(s)"))

        Logger.log("%5d%20.10e" % (self.it_counter, obj), end="")
        for v in foi.values():
            if not isinstance(v, str):
                v = "%20.10e" % v
            Logger.log("%20s" % v, end="")

        fail = 0
        self.it_counter += 1
        return fail, obj, con

    def evalObjConGradient(self, x, g, A):
        t_start = timer()

        if self.dv_mapping is not None:
            # Populate the nodal variable for analysis
            self.xfull[:] = self.dv_mapping.dot(x)  # x = E*xr
            self.xfull[self.non_design_nodes] = 1.0

            if self.objf == "volume":
                g[:] = self.dv_mapping.T.dot(
                    self.analysis.eval_area_gradient(self.xfull)
                )
            elif self.objf == "frequency":
                if self.prob == "natural_frequency":
                    g[:] = self.dv_mapping.T.dot(
                        self.analysis.ks_omega_derivative(self.xfull)
                    )
                elif self.prob == "buckling":
                    g[:] = self.dv_mapping.T.dot(
                        self.analysis.ks_buckling_derivative(self.xfull)
                    )
            elif self.objf == "compliance":
                g[:] = self.dv_mapping.T.dot(
                    self.analysis.compliance_gradient(self.xfull)
                )
            elif self.objf == "displacement":
                g[:] = 10 * self.dv_mapping.T.dot(
                    self.analysis.eigenvector_displacement_deriv(
                        self.xfull, self.ks_rho
                    )
                )
            else:  # objf == "stress"
                g[:] = self.stress_scale * self.dv_mapping.T.dot(
                    self.analysis.eigenvector_stress_derivative(self.xfull)
                )

            # Evaluate constraint gradients
            index = 0
            if "volume_ub" in self.confs:
                A[index][:] = -self.dv_mapping.T.dot(
                    self.analysis.eval_area_gradient(self.xfull)
                )
                index += 1

            if "volume_lb" in self.confs:
                A[index][:] = -self.dv_mapping.T.dot(
                    self.analysis.eval_area_gradient(self.xfull)
                )
                index += 1

            if "frequency" in self.confs:
                if self.prob == "natural_frequency":
                    A[index][:] = self.dv_mapping.T.dot(
                        self.analysis.ks_omega_derivative(self.xfull)
                    )
                elif self.prob == "buckling":
                    A[index][:] = self.dv_mapping.T.dot(
                        self.analysis.ks_buckling_derivative(self.xfull)
                    )
                index += 1

            if "stress" in self.confs:
                A[index][:] = -self.dv_mapping.T.dot(
                    self.analysis.eigenvector_stress_derivative(self.xfull)
                    / self.stress_ub
                )
                index += 1

            if "displacement" in self.confs:
                A[index][:] = -self.dv_mapping.T.dot(
                    self.analysis.eigenvector_displacement_deriv(
                        self.xfull, self.ks_rho
                    )
                )
                index += 1

        else:
            # Populate the nodal variable for analysis
            self.xfull[self.design_nodes] = x[:]

            if self.objf == "volume":
                g[:] = self.analysis.eval_area_gradient(self.xfull)[self.design_nodes]
            elif self.objf == "frequency":
                if self.prob == "natural_frequency":
                    g[:] = (
                        -self.analysis.ks_omega_derivative(self.xfull)[
                            self.design_nodes
                        ]
                    )
                elif self.prob == "buckling":
                    g[:] = (
                        -self.analysis.ks_buckling_derivative(self.xfull)[
                            self.design_nodes
                        ]
                    )
            elif self.objf == "compliance":
                g[:] = self.analysis.compliance_gradient(self.xfull)[self.design_nodes]
            elif self.objf == "displacement":
                g[:] = (
                    10
                    * self.analysis.eigenvector_displacement_deriv(
                        self.xfull, self.ks_rho
                    )[self.design_nodes]
                )
            else:  # objf == "stress"
                g[:] = (
                    self.stress_scale
                    * self.analysis.eigenvector_stress_derivative(self.xfull)[
                        self.design_nodes
                    ]
                )

            # Evaluate constraint gradient
            index = 0
            if "volume_ub" in args.confs:
                A[index][:] = -self.analysis.eval_area_gradient(self.xfull)[
                    self.design_nodes
                ]
                index += 1

            if "volume_lb" in args.confs:
                A[index][:] = -self.analysis.eval_area_gradient(self.xfull)[
                    self.design_nodes
                ]
                index += 1

            if "frequency" in self.confs:
                if self.prob == "natural_frequency":
                    A[index][:] = self.analysis.ks_omega_derivative(self.xfull)[
                        self.design_nodes
                    ]
                elif self.prob == "buckling":
                    A[index][:] = self.analysis.ks_buckling_derivative(self.xfull)[
                        self.design_nodes
                    ]
                index += 1

            if "stress" in self.confs:
                A[index][:] = (
                    -self.analysis.eigenvector_stress_derivative(self.xfull)[
                        self.design_nodes
                    ]
                    / self.stress_ub
                )
                index += 1

            if "displacement" in self.confs:
                A[index][:] = (
                    -self.analysis.eigenvector_displacement_deriv(
                        self.xfull, self.ks_rho
                    )[self.design_nodes]
                    / self.disp_ub
                )
                index += 1

        t_end = timer()
        self.elapse_g = t_end - t_start

        Logger.log("%10.3f%10.3f" % (self.elapse_f, self.elapse_g))

        return 0


try:
    import mma4py

    class MMAProb(mma4py.Problem):
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
    MMAProb = None


class ParOptProb(ParOpt.Problem):
    def __init__(self, comm, prob: TopOptProb) -> None:
        self.prob = prob
        try:
            super().__init__(comm, prob.ndv, prob.ncon)
        except:
            super().__init__(comm, nvars=prob.ndv, ncon=prob.ncon)
        return

    def getVarsAndBounds(self, x, lb, ub):
        return self.prob.getVarsAndBounds(x, lb, ub)

    def evalObjCon(self, x):
        return self.prob.evalObjCon(x)

    def evalObjConGradient(self, x, g, A):
        return self.prob.evalObjConGradient(x, g, A)


def to_vtk(vtk_path, conn, X, nodal_sols={}, cell_sols={}, nodal_vecs={}, cell_vecs={}):
    """
    Generate a vtk given conn, X, and optionally list of nodal solutions

    Args:
        nodal_sols: dictionary of arrays of length nnodes
        cell_sols: dictionary of arrays of length nelems
        nodal_vecs: dictionary of list of components [vx, vy], each has length nnodes
        cell_vecs: dictionary of list of components [vx, vy], each has length nelems
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
        if nodal_sols or nodal_vecs:
            fh.write(f"POINT_DATA {nnodes}\n")

        if nodal_sols:
            for name, data in nodal_sols.items():
                fh.write(f"SCALARS {name} double 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")

        if nodal_vecs:
            for name, data in nodal_vecs.items():
                fh.write(f"VECTORS {name} double\n")
                for val in np.array(data).T:
                    fh.write(f"{val[0]} {val[1]} 0.\n")

        if cell_sols or cell_vecs:
            fh.write(f"CELL_DATA {nelems}\n")

        if cell_sols:
            for name, data in cell_sols.items():
                fh.write(f"SCALARS {name} double 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")

        if cell_vecs:
            for name, data in cell_vecs.items():
                fh.write(f"VECTORS {name} double\n")
                for val in np.array(data).T:
                    fh.write(f"{val[0]} {val[1]} 0.\n")

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


def create_lbracket_domain(r0_=2.1, l=8.0, lfrac=0.4, nx=96, m0_block_frac=0.0):
    """
     _nt__       ________________
    |     |                     ^
    |     |                     |
    |     |_____                l
    |           | lfrac * l     |
    |___________|  _____________|
          nx
    """
    nt = int(nx * lfrac)
    nelems = nx * nx - (nx - nt) * (nx - nt)
    nnodes = (nx + 1) * (nx + 1) - (nx - nt) * (nx - nt)

    nodes_1 = np.arange((nx + 1) * (nt + 1)).reshape(nt + 1, nx + 1)
    nodes_2 = (nx + 1) * (nt + 1) + np.arange((nx - nt) * (nt + 1)).reshape(
        nx - nt, nt + 1
    )

    def ij_to_node(ip, jp):
        if jp <= nt:
            return nodes_1[jp, ip]
        return nodes_2[jp - nt - 1, ip]

    def pt_out_domain(ip, jp):
        return ip > nt and jp > nt

    def elem_out_domain(ie, je):
        return ie >= nt and je >= nt

    X = np.zeros((nnodes, 2))
    index = 0
    for jp in range(nx + 1):  # y-directional index
        for ip in range(nx + 1):  # x-directional index
            if not pt_out_domain(ip, jp):
                X[index, :] = [l / nx * ip, l / nx * jp]
                index += 1

    conn = np.zeros((nelems, 4), dtype=int)
    index = 0
    for je in range(nx):  # y-directional index
        for ie in range(nx):  # x-directional index
            if not elem_out_domain(ie, je):
                conn[index, :] = [
                    ij_to_node(ie, je),
                    ij_to_node(ie + 1, je),
                    ij_to_node(ie + 1, je + 1),
                    ij_to_node(ie, je + 1),
                ]
                index += 1

    non_design_nodes = []
    nm = int(np.ceil(0.1 * nx * m0_block_frac))
    # for jp in range(nt - nm, nt + 1):
    #     for ip in range(nx - nm, nx + 1):
    #         non_design_nodes.append(ij_to_node(ip, jp))

    bcs = {}
    for ip in range(nt + 1):
        bcs[ij_to_node(ip, nx)] = [0, 1]
        # non_design_nodes.append(ij_to_node(ip, nx))
        # non_design_nodes.append(ij_to_node(ip, nx-1))

    offset = int(nx)
    # for j in range(nx):
    #     bcs[ij_to_node(0, j)] = [0, 1]

    for j in range(nt + 1, nx):
        bcs[ij_to_node(nt, j)] = [0]

    # bcs[ij_to_node(0, int(nt/2))] = [0]

    forces = {}
    P = 1.0
    for jp in range(nt, nt + 1):
        for ip in range(nx, nx + 1):
            forces[ij_to_node(ip, jp)] = [0, -P / nm]

    r0 = l / nx * r0_

    return conn, X, r0, bcs, forces, non_design_nodes


def create_beam_domain(r0_=2.1, l=8.0, frac=0.125, nx=100):
    """
    _____________|_____________
    |                         |
    |                         | n
    |_________________________|
                m
    """

    m = nx
    n = int(np.ceil((frac * nx)))

    # make sure m and n is even
    if n % 2 == 0:
        n -= 1
    if m % 2 == 0:
        m -= 1

    nelems = m * n
    nnodes = (m + 1) * (n + 1)

    y = np.linspace(0, l * frac, n + 1)
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

    non_design_nodes = []
    # apply top middle a square block
    # nm = int(np.ceil(2*n * m0_block_frac))
    # for i in range(n - nm+1, n+1):
    #     for j in range((m - nm) // 2 +1, (m + nm) // 2 +1):
    #         non_design_nodes.append(nodes[i, j])

    bcs = {}

    # bcs[nodes[n // 2, 0]] = [0, 1]
    # bcs[nodes[n // 2, m]] = [0, 1]

    # fix the bottom left and right
    offset = int(np.ceil(m * 0.02))
    for i in range(offset):
        bcs[nodes[0, i]] = [0, 1]
        bcs[nodes[0, m - i]] = [0, 1]

    # force is independent of the mesh size
    P = 100.0
    forces = {}
    # apply a force at the top middle
    offset = int(np.ceil(m * 0.01))
    for i in range(offset):
        forces[nodes[n, m // 2 - i]] = [0, -P / (2 * offset)]
        forces[nodes[n, m // 2 + 1 + i]] = [0, -P / (2 * offset)]

    r0 = l / nx * r0_
    ic(r0)

    Ei = []
    Ej = []
    redu_idx = 0

    # 2-way reflection left to right
    for j in range(2 * (n + 1)):
        for i in range((m + 1) // 2):
            if j % 2 == 0:
                Ej.extend([i + j * (m + 1) // 4])
            else:
                Ej.extend([i + (m // 2 - 2 * i) + (j - 1) * (m + 1) // 4])
            Ei.extend([i + j * (m + 1) // 2])

    # # 4-way reflection of x- and y-symmetry axes
    # a = n // 2
    # b = m // 2
    # for i in range(a + 1):
    #     for j in range(b + 1):
    #         if nodes[i, j] not in non_design_nodes:
    #             Ej.extend(4 * [redu_idx])
    #             Ei.extend(
    #                 [
    #                     nodes[i, j],
    #                     nodes[n - i, j],
    #                     nodes[i, m - j],
    #                     nodes[n - i, m - j],
    #                 ]
    #             )
    #             redu_idx += 1

    Ev = np.ones(len(Ei))
    dv_mapping = coo_matrix((Ev, (Ei, Ej)))

    # change dv_mapping to np.array
    # dv_mapping = np.array(dv_mapping.todense())
    # ic(dv_mapping.shape)
    # ic(dv_mapping)

    return conn, X, r0, bcs, forces, non_design_nodes, dv_mapping


def create_building_domain(r0_=2.1, l=8.0, frac=0.125, nx=100, m0_block_frac=0.0):
    """
    _______
    |     |
    |     |
    |     |
    |     |
    |     | n
    |     |
    |     |
    |     |
    |_____|
       m
    """

    m = int(np.ceil((frac * nx)))
    n = nx

    # make sure number of nodes along x is even
    if m % 2 == 0:
        m += 1

    nelems = m * n
    nnodes = (m + 1) * (n + 1)

    y = np.linspace(0, l, n + 1)
    x = np.linspace(0, l * frac, m + 1)
    nodes = np.arange(0, (n + 1) * (m + 1)).reshape((n + 1, m + 1))

    ic(nodes.shape)
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

    non_design_nodes = []
    # apply top middle a square block
    nm = int(np.ceil(m * m0_block_frac))
    if nm % 2 == 1:
        nm -= 1
    nm = 2

    # for i in range(n - nm + 1, n + 1):
    #     for j in range((m - nm) // 2 + 1, (m + nm) // 2 + 1):
    #         non_design_nodes.append(nodes[i, j])

    # for i in range(n - nm + 1, n + 1):
    #     for j in range(0, nm):
    #         non_design_nodes.append(nodes[i, j])
    # for i in range(n - nm + 1, n + 1):
    #     for j in range(m - nm + 1, m + 1):
    #         non_design_nodes.append(nodes[i, j])

    # for i in range(n - nm + 1, n + 1):
    #     for j in range(0, m + 1):
    #         non_design_nodes.append(nodes[i, j])

    # h = n // 8
    # for i in range(1, 9):
    #     for j in range(m + 1):
    #         non_design_nodes.append(nodes[i * h, j])

    bcs = {}

    offset = int(np.floor(n * 0.0002))
    for j in range(m + 1):
        bcs[nodes[0, j]] = [0, 1]

    # bcs[nodes[0, 0]] = [0, 1]
    # bcs[nodes[0, m]] = [0, 1]

    # force is independent of the mesh size
    P = 100.0
    forces = {}
    # # apply a force at the top middle
    # offset = int(np.ceil(m * 0.01))
    # for i in range(offset):
    #     forces[nodes[n, m // 2 - i]] = [0, -P / (2 * offset)]
    #     forces[nodes[n, m // 2 + 1 + i]] = [0, -P / (2 * offset)]
    forces[nodes[n, m // 2]] = [P, 0]

    r0 = l / nx * r0_
    ic(r0)

    Ei = []
    Ej = []

    # 2-way reflection left to right
    for j in range(2 * (n + 1)):
        for i in range((m + 1) // 2):
            if j % 2 == 0:
                Ej.extend([i + j * (m + 1) // 4])
            else:
                Ej.extend([i + (m // 2 - 2 * i) + (j - 1) * (m + 1) // 4])
            Ei.extend([i + j * (m + 1) // 2])

    Ev = np.ones(len(Ei))
    dv_mapping = coo_matrix((Ev, (Ei, Ej)))

    # change dv_mapping to np.array
    # dv_mapping = np.array(dv_mapping.todense())
    # ic(dv_mapping.shape)
    # ic(dv_mapping)

    return conn, X, r0, bcs, forces, non_design_nodes, dv_mapping


def create_square_domain(r0_, l=8.0, nx=30, m0_block_frac=0.0):
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
    # offset = int(m0_block_frac * nx * 0.5)
    # for j in range(n // 2 - offset, (n + 1) // 2 + 1 + offset):
    #     for i in range(n // 2 - offset, (n + 1) // 2 + 1 + offset):
    #         non_design_nodes.append(nodes[j, i])

    # Constrain all boundaries
    bcs = {}

    offset = int(nx * 0.1)

    for i in range(offset):
        bcs[nodes[0, i]] = [1]
        bcs[nodes[0, m - i]] = [1]
        bcs[nodes[n, i]] = [1]
        bcs[nodes[n, m - i]] = [1]

    for j in range(offset):
        bcs[nodes[j, 0]] = [0]
        bcs[nodes[j, m]] = [0]
        bcs[nodes[n - j, 0]] = [0]
        bcs[nodes[n - j, m]] = [0]

    # fix the bottom left corner
    bcs[nodes[0, 0]] = [0, 1]
    # fix the bottom right corner
    bcs[nodes[0, m]] = [0, 1]
    # fix the top left corner
    bcs[nodes[n, 0]] = [0, 1]
    # fix the top right corner
    bcs[nodes[n, m]] = [0, 1]

    # P = 10.0
    forces = {}
    # pn = n // 10
    # for j in range(pn):
    #     forces[nodes[j, -1]] = [0, -P / pn]

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


def visualize_domain(prefix, X, bcs, non_design_nodes=None, forces=None):
    markersize = 1.0
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    ax.scatter(X[:, 0], X[:, 1], color="black", s=markersize)

    if bcs:
        for i, v in bcs.items():
            if len(v) == 2:
                ax.scatter(X[i, 0], X[i, 1], color="red", s=markersize)
            else:
                ax.scatter(X[i, 0], X[i, 1], color="g", s=markersize)

    if forces:
        for i, v in forces.items():
            ax.scatter(X[i, 0], X[i, 1], color="orange", s=markersize)

    if non_design_nodes:
        m0_X = np.array([X[i, :] for i in non_design_nodes])
        ax.scatter(m0_X[:, 0], m0_X[:, 1], color="blue", s=markersize)

    fig.savefig(os.path.join(prefix, "domain.png"), dpi=500, bbox_inches="tight")
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
        "mma_init_asymptote_offset": 0.2,
        "max_major_iters": 100,
        "penalty_gamma": 1e3,
        "qn_subspace_size": 10,
        "qn_type": "bfgs",
        "abs_res_tol": 1e-8,
        "starting_point_strategy": "affine_step",
        # "barrier_strategy": "mehrotra_predictor_corrector",
        "barrier_strategy": "mehrotra",
        "use_line_search": False,
        # "mma_constraints_delta": True,
        # "mma_move_limit": 0.2,
        "output_file": os.path.join(prefix, "paropt.out"),
        "tr_output_file": os.path.join(prefix, "paropt.tr"),
        "mma_output_file": os.path.join(prefix, "paropt.mma"),
    }
    return options


def parse_cmd_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # OS
    p.add_argument("--prefix", default="output", type=str, help="output folder")

    # Analysis
    p.add_argument(
        "--domain",
        default="square",
        choices=["square", "beam", "lbracket", "building"],
    )
    p.add_argument(
        "--problem",
        default="natural_frequency",
        choices=["natural_frequency", "buckling"],
    )
    p.add_argument(
        "--assume-same-element",
        action="store_true",
        help="assume all elements have identical shape",
    )
    p.add_argument(
        "--nx", default=48, type=int, help="number of elements along x direction"
    )
    p.add_argument(
        "--m0-block-frac",
        default=0.0,
        type=float,
        help="fraction of the size of non-design mass block with respect to the domain",
    )
    p.add_argument(
        "--stress-relax", default=0.3, type=float, help="stress relaxation factor"
    )
    p.add_argument(
        "--filter",
        default="spatial",
        choices=["spatial", "helmholtz"],
        help="density filter type",
    )
    p.add_argument("--ks-rho", default=100, type=int, help="ks aggregation parameter")
    p.add_argument(
        "--ptype-K",
        default="simp",
        choices=["ramp", "simp"],
        help="material penalization for stiffness matrix",
    )
    p.add_argument(
        "--ptype-M",
        default="linear",
        choices=["ramp", "msimp", "linear"],
        help="material penalization for stiffness matrix",
    )
    p.add_argument("--p", default=3.0, type=float, help="SIMP penalization parameter")
    p.add_argument("--q", default=5.0, type=float, help="RAMP penalization parameter")
    p.add_argument(
        "--rho0-K", default=1e-3, type=float, help="rho offset to prevent singular K"
    )
    p.add_argument(
        "--rho0-M", default=1e-7, type=float, help="rho offset to prevent singular M"
    )

    p.add_argument("--m0", default=0.0, type=float, help="magnitude of non-design mass")
    p.add_argument("--r0", default=2.1, type=float, help="filter radius = r0 * lx / nx")

    # Optimization
    p.add_argument(
        "--optimizer",
        default="pmma",
        choices=["pmma", "mma4py", "tr"],
        help="optimization method",
    )
    p.add_argument(
        "--movelim",
        default=0.2,
        type=float,
        help="move limit for design variables, for mma4py only",
    )
    p.add_argument(
        "--lb", default=1e-06, type=float, help="lower bound of design variables"
    )
    p.add_argument(
        "--objf",
        default="frequency",
        choices=["frequency", "stress", "volume", "compliance", "displacement"],
        help="objective function",
    )
    p.add_argument(
        "--confs",
        default="volume_ub",
        nargs="*",
        choices=["volume_ub", "volume_lb", "frequency", "stress", "displacement"],
        help="constraint functions",
    )
    p.add_argument(
        "--omega-lb",
        default=None,
        type=float,
        help='Lower bound of natural frequency, only effective when "frequency" is in the confs',
    )
    p.add_argument(
        "--stress-ub",
        default=None,
        type=float,
        help='Upper bound for stress constraint, only effective when "stress" is in the confs',
    )
    p.add_argument(
        "--vol-frac-ub",
        default=None,
        type=float,
        help='volume fraction, only effective when "volume_ub" is in the confs',
    )
    p.add_argument(
        "--vol-frac-lb",
        default=None,
        type=float,
        help='volume fraction, only effective when "volume_lb" is in the confs',
    )
    p.add_argument(
        "--stress-scale",
        default=1e-9,
        type=float,
        help='scale the stress objective obj = stress * scale, only effective when objf is "stress"',
    )
    p.add_argument(
        "--dis-ub",
        default=None,
        type=float,
        help='displacement constraint, only effective when "displacement" is in the confs',
    )
    p.add_argument(
        "--mode",
        default=1,
        type=int,
        help='mode number, only effective when "displacement" is in the confs',
    )
    p.add_argument(
        "--maxit", default=200, type=int, help="maximum number of iterations"
    )
    p.add_argument(
        "--grad-check", action="store_true", help="perform gradient check and exit"
    )
    p.add_argument(
        "--proj",
        default=False,
        type=bool,
        help="projector for filter",
    )
    p.add_argument(
        "--note",
        default="",
        type=str,
        help="note for the run",
    )
    args = p.parse_args()

    return args


def create_folder(args):
    if not os.path.isdir(args.prefix):
        os.mkdir(args.prefix)
    if args.confs == ["stress", "frequency"]:
        args.confs = ["frequency", "stress"]
    if args.confs == ["frequency", "volume_ub"]:
        args.confs = ["volume_ub", "frequency"]
    if args.confs == ["stress", "volume_ub"]:
        args.confs = ["volume_ub", "stress"]
    if args.confs == ["volume_ub", "volume_lb"]:
        args.confs = ["volume_lb", "volume_ub"]
    if args.confs == ["volume_ub", "displacement"]:
        args.confs = ["displacement", "volume_ub"]

    name = f"{args.domain}"
    if not os.path.isdir(os.path.join(args.prefix, name)):
        os.mkdir(os.path.join(args.prefix, name))
    args.prefix = os.path.join(args.prefix, name)

    # make a folder inside each domain folder to store the results of each run
    name2 = f"{args.objf}{args.confs}"
    if not os.path.isdir(os.path.join(args.prefix, name2)):
        os.mkdir(os.path.join(args.prefix, name2))
    args.prefix = os.path.join(args.prefix, name2)

    o = f"{args.optimizer[0]}"
    args.prefix = os.path.join(args.prefix, o)

    if args.domain == "square":
        if args.nx != 400:
            n = f"{args.nx}"
            args.prefix = args.prefix + ", n=" + n
    elif args.domain == "beam":
        if args.nx != 800:
            n = f"{args.nx}"
            args.prefix = args.prefix + ", n=" + n

    if args.confs != []:
        if args.domain == "square":
            if args.vol_frac_ub != 0.4:
                v = f"{args.vol_frac_ub:.1f}"
                args.prefix = args.prefix + ", v=" + v
        elif args.domain == "beam":
            if args.vol_frac_ub != 0.5:
                v = f"{args.vol_frac_ub:.1f}"
                args.prefix = args.prefix + ", v=" + v

    if args.m0_block_frac != 0.0:
        m = f"{args.m0_block_frac:.2f}"
        args.prefix = args.prefix + ", m0=" + m

    if "displacement" in args.confs:
        if args.mode != 1:
            args.prefix = args.prefix + ", mode=" + str(args.mode)
        d = f"{args.dis_ub:.3f}"
        args.prefix = args.prefix + ", d=" + d

    if "stress" in args.confs:
        s = f"{args.stress_ub/ 1e12}"
        args.prefix = args.prefix + ", s=" + s

    if args.r0 != 2.1:
        r = f"{args.r0}"
        args.prefix = args.prefix + ", r=" + r

    if args.note != "":
        args.prefix = args.prefix + ", " + args.note

    # if args.dis_ub is not None:
    #     d = f"{args.dis_ub:.3f}"
    #     args.prefix = os.path.join(args.prefix, d)

    # if args.stress_ub is not None:
    #     s = f"{args.stress_ub/ 1e12}"
    #     args.prefix = os.path.join(args.prefix, s)

    # # create a folder inside the result folder to store the results of each run
    # if args.confs == ["volume_ub", "stress"]:
    #     args.prefix = os.path.join(
    #         args.prefix,
    #         o + ", n=" + n + ", s=" + s + args.note,
    #     )
    # elif args.confs == ["volume_ub"]:
    #     args.prefix = os.path.join(
    #         args.prefix,
    #         o + ", n=" + n + args.note,
    #     )
    # elif args.confs == ["displacement", "volume_ub"]:
    #     args.prefix = os.path.join(
    #         args.prefix,
    #         o + ", n=" + n + ", d=" + d + args.note,
    #     )
    # elif args.confs == ["volume_ub", "displacement", "stress"]:
    #     args.prefix = os.path.join(
    #         args.prefix,
    #         o + ", n=" + n + ", d=" + d + ", s=" + s + args.note,
    #     )

    if os.path.isdir(args.prefix):
        rmtree(args.prefix)
    os.mkdir(args.prefix)

    return args


def main(args):
    args = create_folder(args)
    # Set up logger
    Logger.set_log_path(os.path.join(args.prefix, "stdout.log"))
    timer_set_log_path(os.path.join(args.prefix, "profiler.log"))

    # Save option values
    with open(os.path.join(args.prefix, "options.txt"), "w") as f:
        f.write("Options:\n")
        for k, v in vars(args).items():
            f.write(f"{k:<20}{v}\n")

    if args.domain == "square":
        conn, X, r0, bcs, forces, non_design_nodes, dv_mapping = create_square_domain(
            r0_=args.r0, nx=args.nx, m0_block_frac=args.m0_block_frac
        )
    elif args.domain == "lbracket":
        conn, X, r0, bcs, forces, non_design_nodes = create_lbracket_domain(
            r0_=args.r0, nx=args.nx, m0_block_frac=args.m0_block_frac
        )
        dv_mapping = None
    elif args.domain == "beam":
        conn, X, r0, bcs, forces, non_design_nodes, dv_mapping = create_beam_domain(
            r0_=args.r0, nx=args.nx
        )
    elif args.domain == "building":
        (
            conn,
            X,
            r0,
            bcs,
            forces,
            non_design_nodes,
            dv_mapping,
        ) = create_building_domain(
            r0_=args.r0, nx=args.nx, m0_block_frac=args.m0_block_frac
        )

    # Check the mesh
    visualize_domain(args.prefix, X, bcs, non_design_nodes, forces)

    # for there is displacement constraint, we need to use the displacement constraint
    if "beam" in args.domain:
        m = args.nx
        n = int(np.ceil((0.125 * args.nx)))
    elif "building" in args.domain:
        m = int(np.ceil((0.125 * args.nx)))
        n = args.nx
    elif "square" in args.domain:
        m = args.nx
        n = args.nx
    else:
        raise ValueError("Not supported domain")

    D_index = None
    ic(args.domain, args.objf, args.confs, m, n)
    for conf in args.confs:
        if conf == "displacement":
            if args.domain == "beam":
                if args.mode == 1:
                    # node_loc=(n, m * 0.5), y direction
                    D_index = 2 * int((n * m - m * 0.5)) + 1
                elif args.mode == 2:
                    # node_loc=(n, m * 0.5), y direction
                    D_index = 2 * (m // 2 - 1) + 1
                else:
                    # node_loc=(n * 0.5, m * 0.5), y direction
                    D_index = n * m - 1
            if args.domain == "square":
                if args.mode == 1:
                    # node_loc=(0.75*n, 0.75*m), x direction
                    indx = int((0.75 * n * (m + 1) + 0.75 * m))
                    ic(indx)
                    D_index = [2 * indx, 2 * indx + 1]
                elif args.mode == 2:
                    indx = int((0.67 * n * (m + 1) + 0.67 * m))
                    D_index = [2 * indx, 2 * indx + 1]

    # Create the filter
    fltr = NodeFilter(conn, X, r0, ftype=args.filter, projection=args.proj)
    # Create analysis
    analysis = TopologyAnalysis(
        fltr,
        conn,
        X,
        bcs,
        forces,
        m,
        D_index,
        ptype_K=args.ptype_K,
        ptype_M=args.ptype_M,
        rho0_K=args.rho0_K,
        rho0_M=args.rho0_M,
        p=args.p,
        q=args.q,
        epsilon=args.stress_relax,
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
        prob=args.problem,
        objf=args.objf,
        confs=args.confs,
        omega_lb=args.omega_lb,
        stress_ub=args.stress_ub,
        vol_frac_ub=args.vol_frac_ub,
        vol_frac_lb=args.vol_frac_lb,
        stress_scale=args.stress_scale,
        dis_ub=args.dis_ub,
        grad_check=args.grad_check,
        domain=args.domain,
    )

    # Print info
    Logger.log("=== Problem overview ===")
    Logger.log("objective:   %s" % args.objf)
    Logger.log(f"constraints: {args.confs}")
    Logger.log("num of dof:  %d" % analysis.nvars)
    Logger.log("num of dv:   %d" % topo.ndv)
    Logger.log()

    if args.optimizer == "mma4py":
        if MMAProb is None:
            raise ImportError("Cannot use mma4py, package not found.")

        mmaprob = MMAProb(topo)
        mmaopt = mma4py.Optimizer(
            mmaprob, log_name=os.path.join(args.prefix, "mma4py.log")
        )
        if args.grad_check:
            np.random.seed(0)
            for i in range(5):
                mmaopt.checkGradients()
            exit(0)

        mmaopt.optimize(niter=args.maxit, verbose=False, movelim=args.movelim)
        xopt = mmaopt.getOptimizedDesign()

    else:
        from mpi4py import MPI

        paroptprob = ParOptProb(MPI.COMM_SELF, topo)

        # if args.grad_check:
        for i in range(5):
            paroptprob.checkGradients(1e-6)
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
        xopt, z, zw, zl, zu = opt.getOptimizedPoint()

    # Evaluate the stress for optimzied design
    topo.evalObjCon(xopt, eval_all=True)


if __name__ == "__main__":
    args = parse_cmd_args()
    main(args)
