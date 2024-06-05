from collections import OrderedDict
import os
import shutil as shutil
import sys
import time
from timeit import default_timer as timer

from icecream import ic
import matplotlib.pylab as plt
import matplotlib.tri as tri
import mpmath as mp
import numpy as np
from paropt import ParOpt
from scipy import sparse, spatial
from scipy.linalg import cholesky, eigh, eigvalsh, qr, solve
from scipy.sparse import coo_matrix, linalg

import domain as domain
import other.utils as utils
from other.utils import time_this, timer_set_log_path
import settings as settings

# from other.test_lobpcg import lobpcg4

# numpy.set_printoptions(threshold=sys.maxsize)


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
        self, conn, X, r0=1.0, ftype="spatial", beta=2.0, eta=0.5, projection=False
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
        D_index=None,
        fun="tanh",
        N_a=0,
        N_b=10,
        N=12,
        atype=True,
        E=1.0,  # to make sure eigs is not too large, E=10.0 * 1e6,
        nu=0.3,
        ptype_K="ramp",
        ptype_M="ramp",
        rho0_K=1e-3,
        rho0_M=1e-7,
        rho0_G=0.0,
        p=3.0,
        q=5.0,
        density=1.0,
        epsilon=0.3,
        assume_same_element=False,
        check_gradient=False,
        check_kokkos=False,
        prob="natural_frequency",
        kokkos=False,
    ):
        self.ptype_K = ptype_K.lower()
        self.ptype_M = ptype_M.lower()

        self.rho0_K = rho0_K
        self.rho0_M = rho0_M
        self.rho0_G = rho0_G

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

        self.D_index = D_index.astype(int)
        self.K0 = None
        self.M0 = None

        self.fun = getattr(mp, fun)
        self.N_a = N_a
        self.N_b = N_b
        self.N = N
        self.atype = atype

        self.check_gradient = check_gradient
        self.check_kokkos = check_kokkos
        self.prob = prob
        self.kokkos = kokkos

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

        if not self.kokkos or self.check_kokkos:
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
                C = np.outer(
                    rhoE / (1.0 + self.q * (1.0 - rhoE)) + self.rho0_K, self.C0
                )

            C = C.reshape((self.nelems, 3, 3))

            # Compute the element stiffness matrix
            gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

            # Assemble all of the the 8 x 8 element stiffness matrix
            Ke_python = np.zeros((self.nelems, 8, 8), dtype=rho.dtype)

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
                        Ke_python += detJ * np.einsum("ij,nik,kl -> njl", Be_, C, Be_)

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
                        Ke_python += np.einsum("n,nij,nik,nkl -> njl", detJ, Be, C, Be)
                Ke_python = Ke_python.flatten()

        if self.kokkos or self.check_kokkos:
            import kokkos as kokkos

            Ke_kokkos = kokkos.assemble_stiffness_matrix(
                rho,
                self.detJ,
                self.Be,
                self.conn,
                self.C0,
                self.rho0_K,
                self.ptype_K,
                self.p,
                self.q,
            )

        if self.check_kokkos:
            ic(np.allclose(Ke_python, Ke_kokkos))

        if not self.kokkos:
            Ke = Ke_python
        else:
            Ke = Ke_kokkos

        K = sparse.coo_matrix((Ke, (self.i, self.j)))
        K = K.tocsr()

        if self.K0 is not None:
            K += self.K0

        return K

    @time_this
    def stiffness_matrix_derivative(self, rho, psi, u):
        """
        Compute the derivative of the stiffness matrix times the vectors psi and u
        """
        # if don't have the kokkos version, use the python version
        if not self.kokkos or self.check_kokkos:
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

                        dfdC += detJ * np.einsum(
                            "im,jl,nm,nl -> nij", Be_, Be_, psie, ue
                        )

            else:
                Be = np.zeros((self.nelems, 3, 8))
                xe = self.X[self.conn, 0]
                ye = self.X[self.conn, 1]

                for j in range(2):
                    for i in range(2):
                        xi = gauss_pts[i]
                        eta = gauss_pts[j]
                        detJ = _populate_Be(self.nelems, xi, eta, xe, ye, Be)

                        dfdC += np.einsum(
                            "n,nim,njl,nm,nl -> nij", detJ, Be, Be, psie, ue
                        )

            dfdrhoE = np.zeros(self.nelems)
            for i in range(3):
                for j in range(3):
                    dfdrhoE[:] += self.C0[i, j] * dfdC[:, i, j]

            if self.ptype_K == "simp":
                dfdrhoE[:] *= self.p * rhoE ** (self.p - 1.0)
            else:  # ramp
                dfdrhoE[:] *= (1.0 + self.q) / (1.0 + self.q * (1.0 - rhoE)) ** 2

            dKdrho_python = np.zeros(self.nnodes)
            for i in range(4):
                np.add.at(dKdrho_python, self.conn[:, i], dfdrhoE)
            dKdrho_python *= 0.25

        if self.kokkos or self.check_kokkos:
            import kokkos as kokkos

            dKdrho_kokkos = kokkos.stiffness_matrix_derivative(
                rho,
                self.detJ,
                self.Be,
                self.conn,
                u,
                psi,
                self.C0,
                self.ptype_K,
                self.p,
                self.q,
            )

        if self.check_kokkos:
            ic(np.allclose(dKdrho_python, dKdrho_kokkos))

        if not self.kokkos:
            dKdrho = dKdrho_python
        else:
            dKdrho = dKdrho_kokkos

        return dKdrho

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
        Me = np.zeros((self.nelems, 8, 8), dtype=rho.dtype)

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
            dnonlin = 6.0 * self.simp_c1 * rhoE**5.0 + 7.0 * self.simp_c2 * rhoE**6.0
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
        if not self.kokkos or self.check_kokkos:
            # Average the density to get the element-wise density
            rhoE = 0.25 * (
                rho[self.conn[:, 0]]
                + rho[self.conn[:, 1]]
                + rho[self.conn[:, 2]]
                + rho[self.conn[:, 3]]
            )

            # Get the element-wise solution variables
            ue = np.zeros((self.nelems, 8), dtype=rho.dtype)
            ue[:, ::2] = u[2 * self.conn]
            ue[:, 1::2] = u[2 * self.conn + 1]

            # Compute the element stiffnesses
            if self.ptype_K == "simp":
                C = np.outer(rhoE**self.p + self.rho0_G, self.C0)
            else:  # ramp
                C = np.outer(
                    rhoE / (1.0 + (self.q + 1) * (1.0 - rhoE)) + self.rho0_G, self.C0
                )

            C = C.reshape((self.nelems, 3, 3))

            # Compute the element stiffness matrix
            gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]

            # Assemble all of the the 8 x 8 element stiffness matrix
            Ge_python = np.zeros((self.nelems, 8, 8), dtype=rho.dtype)

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
                        Ge_python[:, 0::2, 0::2] += G0e
                        Ge_python[:, 1::2, 1::2] += G0e
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
                        Ge_python[:, 0::2, 0::2] += G0e
                        Ge_python[:, 1::2, 1::2] += G0e

                Ge_python = Ge_python.flatten()

        if self.kokkos or self.check_kokkos:
            import kokkos as kokkos

            Ge_kokkos = kokkos.assemble_stress_stiffness(
                rho,
                u,
                self.detJ,
                self.Be,
                self.Te,
                self.conn,
                self.C0,
                self.rho0_G,
                self.ptype_K,
                self.p,
                self.q,
            )

        if self.check_kokkos:
            ic(np.allclose(Ge_python, Ge_kokkos))

        if not self.kokkos:
            Ge = Ge_python
        else:
            Ge = Ge_kokkos

        # Assemble the global stiffness matrix
        G = sparse.coo_matrix((Ge, (self.i, self.j)))
        G = G.tocsr()

        return G

    @time_this
    def stress_stiffness_derivative(self, rho, u, psi, phi, solver):
        """
        Compute the derivative of psi^{T} * G(u, x) * phi using the adjoint method.

        Note "solver" returns the solution of the system of equations

        K * sol = rhs

        Given the right-hand-side rhs. ie. sol = solver(rhs)
        """

        if not self.kokkos or self.check_kokkos:
            # Average the density to get the element-wise density
            rhoE_python = 0.25 * (
                rho[self.conn[:, 0]]
                + rho[self.conn[:, 1]]
                + rho[self.conn[:, 2]]
                + rho[self.conn[:, 3]]
            )

            dfdC_python = np.zeros((self.nelems, 3, 3))

            # Compute the element stiffnesses
            if self.ptype_K == "simp":
                C = np.outer(rhoE_python**self.p + self.rho0_G, self.C0)
            else:  # ramp
                C = np.outer(
                    rhoE_python / (1.0 + (self.q + 1) * (1.0 - rhoE_python))
                    + self.rho0_G,
                    self.C0,
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
            phie[:, ::2] = phi[2 * self.conn]
            phie[:, 1::2] = phi[2 * self.conn + 1]

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
                        # s = np.einsum("nij,njk,nk -> ni", C, Be, ue)

                        # Compute the derivative of the stress w.r.t. u
                        dfds = np.einsum(
                            "n,nijl,nj,nl -> ni", detJ, Te, psie[:, ::2], phie[:, ::2]
                        ) + np.einsum(
                            "n,nijl,nj,nl -> ni", detJ, Te, psie[:, 1::2], phie[:, 1::2]
                        )

                        # Add up contributions to d( psi^{T} * G(x, u) * phi ) / du
                        dfdue += np.einsum("nij,nik,nk -> nj", Be, C, dfds)

                        # Add contributions to the derivative w.r.t. C
                        dfdC_python += np.einsum("ni,njk,nk -> nij", dfds, Be, ue)

            dfdu_python = np.zeros(2 * self.nnodes)
            np.add.at(dfdu_python, 2 * self.conn, dfdue[:, 0::2])
            np.add.at(dfdu_python, 2 * self.conn + 1, dfdue[:, 1::2])

        if self.kokkos or self.check_kokkos:
            import kokkos as kokkos

            rhoE_kokkos, dfdu_kokkos, dfdC_kokkos = kokkos.stress_stiffness_derivative(
                rho,
                u,
                self.detJ,
                self.Be,
                self.Te,
                self.conn,
                psi,
                phi,
                self.C0,
                self.rho0_G,
                self.ptype_K,
                self.p,
                self.q,
            )

            dfdC_kokkos = np.reshape(dfdC_kokkos, (self.nelems, 3, 3))

        if self.check_kokkos:
            ic(np.allclose(rhoE_python, rhoE_kokkos))
            ic(np.allclose(dfdu_python, dfdu_kokkos))
            ic(np.allclose(dfdC_python, dfdC_kokkos))

        if not self.kokkos:
            rhoE = rhoE_python
            dfdu = dfdu_python
            dfdC = dfdC_python
        else:
            rhoE = rhoE_kokkos
            dfdu = dfdu_kokkos
            dfdC = dfdC_kokkos

        # convert self.reduced to a numpy array
        # self.reduced = np.array(self.reduced)

        dfdur = self.reduce_vector(dfdu)

        # Compute the adjoint for K * adj = d ( psi^{T} * G(u, x) * phi ) / du
        adjr = solver(dfdur)
        adj = self.full_vector(adjr)

        dfdrhoE = np.zeros(self.nelems)
        for i in range(3):
            for j in range(3):
                dfdrhoE[:] += self.C0[i, j] * dfdC[:, i, j]

        if self.ptype_K == "simp":
            dfdrhoE[:] *= self.p * rhoE ** (self.p - 1)
        else:  # ramp
            dfdrhoE[:] *= (2.0 + self.q) / (1.0 + (self.q + 1.0) * (1.0 - rhoE)) ** 2

        dfdrho = np.zeros(self.nnodes)

        for i in range(4):
            np.add.at(dfdrho, self.conn[:, i], dfdrhoE)
        dfdrho *= 0.25

        # Compute the derivative of the stiffness matrix w.r.t. rho
        dfdrho -= self.stiffness_matrix_derivative(rho, adj, u)

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
        temp = np.zeros(self.nvars, dtype=vec.dtype)
        temp[self.reduced] = vec[:]
        return temp

    def full_matrix(self, mat):
        """
        Transform from a reduced matrix without dirichlet BCs to the full matrix
        """
        temp = np.zeros((self.nvars, self.nvars), dtype=mat.dtype)
        for i in range(len(self.reduced)):
            for j in range(len(self.reduced)):
                temp[self.reduced[i], self.reduced[j]] = mat[i, j]
        return temp

    def solve(self, x):
        """
        Perform a linear static analysis
        """

        fr = self.reduce_vector(self.f)

        # Method 1: use the factorized matrix
        # Kfact = linalg.factorized(self.Kr)
        # ur = Kfact(fr)
        # u = self.full_vector(ur)

        # Method 2: use the sparse solver
        ur = sparse.linalg.spsolve(self.Kr, fr)
        u = self.full_vector(ur)

        return u

    def compliance(self, x):
        if self.u is None:
            self.u = self.solve(x)
        return self.f.dot(self.u)

    def compliance_gradient(self, x):
        dfdrho = -1.0 * self.stiffness_matrix_derivative(self.rho, self.u, self.u)
        return self.fltr.applyGradient(dfdrho, x)

    def eval_area(self, x):
        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            self.rho[self.conn[:, 0]]
            + self.rho[self.conn[:, 1]]
            + self.rho[self.conn[:, 2]]
            + self.rho[self.conn[:, 3]]
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

    def softmax_ab(self, fun, rho, lam, lam_a, lam_b):
        """
        Compute the eta values
        """
        eta = np.zeros(len(lam), dtype=lam.dtype)
        for i in range(len(lam)):
            a = fun(rho * (lam[i] - lam_a))
            b = fun(rho * (lam[i] - lam_b))
            eta[i] = a - b
        return eta

    @time_this
    def solve_eigenvalue_problem(
        self, x, ks_rho=1000.0, sigma=0.1, nodal_sols=None, nodal_vecs=None, solve=True
    ):
        """
        Compute the k-th smallest natural frequencies
        Populate nodal_sols and cell_sols if provided
        """
        if ks_rho > 1000.0:
            ks_rho = 1000.0

        if self.N > len(self.reduced):
            self.N = len(self.reduced)

        # Compute the density at each node
        self.rho = self.fltr.apply(x)

        # Compute detJ, Be, and Te
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        self.detJ = np.zeros((2, 2, self.nelems))
        self.Be = np.zeros((2, 2, self.nelems, 3, 8))
        self.Be00 = np.zeros((self.nelems, 3, 8))
        self.Te = np.zeros((2, 2, self.nelems, 3, 4, 4))

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                if not self.kokkos or self.check_kokkos:
                    Be_python = np.zeros((self.nelems, 3, 8))
                    Te_python = np.zeros((self.nelems, 3, 4, 4))
                    detJ_python = _populate_Be_and_Te(
                        self.nelems, xi, eta, xe, ye, Be_python, Te_python
                    )
                    if i == 0 and j == 0:
                        _populate_Be(self.nelems, 0.0, 0.0, xe, ye, self.Be00)

                if self.kokkos or self.check_kokkos:
                    import kokkos as kokkos

                    detJ_kokkos, Be_kokkos, Te_kokkos = kokkos.populate_Be_Te(
                        xi, eta, xe, ye
                    )
                    Be_kokkos = Be_kokkos.reshape((self.nelems, 3, 8))
                    Te_kokkos = Te_kokkos.reshape((self.nelems, 3, 4, 4))
                    if i == 0 and j == 0:
                        _, self.Be00 = kokkos.populate_Be(0.0, 0.0, xe, ye)
                        self.Be00 = self.Be00.reshape((self.nelems, 3, 8))

                if self.check_kokkos:
                    ic(np.allclose(detJ_python, detJ_kokkos))
                    ic(np.allclose(Be_python, Be_kokkos))
                    ic(np.allclose(Te_python, Te_kokkos))

                if not self.kokkos:
                    self.detJ[i, j, :] = detJ_python
                    self.Be[i, j, :, :, :] = Be_python
                    self.Te[i, j, :, :, :, :] = Te_python
                else:
                    self.detJ[i, j, :] = detJ_kokkos
                    self.Be[i, j, :, :, :] = Be_kokkos
                    self.Te[i, j, :, :, :, :] = Te_kokkos

        K = self.assemble_stiffness_matrix(self.rho)
        Kr = self.reduce_matrix(K)

        M = self.assemble_mass_matrix(self.rho)
        Mr = self.reduce_matrix(M)
        Mfact = linalg.factorized(Mr)

        # Find the eigenvalues closest to zero. This uses a shift and
        # invert strategy around sigma = 0, which means that the largest
        # magnitude values are closest to zero.
        if solve:
            if self.N == len(self.reduced):
                eigs, Qr = eigh(Kr.todense(), Mr.todense())
            else:
                eigs, Qr = sparse.linalg.eigsh(
                    Kr, M=Mr, k=self.N, sigma=sigma, which="LM", tol=1e-10
                )

            ic(np.allclose(Kr @ Qr, Mr @ Qr @ np.diag(eigs), atol=1e-10))
        else:
            eigs = np.zeros(self.N)
            Qr = np.zeros((Kr.shape[0], self.N))

        Q = np.zeros((self.nvars, self.N), dtype=self.rho.dtype)
        for i in range(self.N):
            Q[self.reduced, i] = Qr[:, i]

        # Save vtk output data
        if nodal_sols is not None:
            nodal_sols["x"] = np.array(x)
            nodal_sols["rho"] = np.array(self.rho)

        if nodal_vecs is not None:
            for i in range(self.N):
                nodal_vecs["phi%d" % i] = [Q[0::2, i], Q[1::2, i]]

        self.A = K
        self.B = M
        self.Kr = Kr

        self.eigs = eigs
        self.Q = Q

        self.u = None
        self.Bfact = Mfact

        # defind dA(q1, q2) as a function of q1 and q2 and can be called from the outside
        self.dA = lambda q1, q2: self.stiffness_matrix_derivative(self.rho, q1, q2)
        self.dB = lambda q1, q2: self.mass_matrix_derivative(self.rho, q1, q2)
        self.lam = self.eigs  # Ar @ Qr = lam * Br @ Qr

        self.lam_a = self.lam[self.N_a] - np.min(np.abs(self.lam)) * 0.5
        self.lam_b = self.lam[self.N_b] + np.min(np.abs(self.lam)) * 0.5

        if self.atype:
            self.eta = np.exp(-ks_rho * (self.lam - np.min(self.lam)))
            self.ks_rho = ks_rho
            self.N_a = 0
            self.N_b = self.N - 1
        else:
            if self.fun == mp.exp:
                self.ks_rho = -ks_rho
                self.eta = -self.softmax_ab(
                    self.fun, self.ks_rho, self.lam, self.lam_a, self.lam_b
                )
            else:
                self.ks_rho = ks_rho
                self.eta = self.softmax_ab(
                    self.fun, self.ks_rho, self.lam, self.lam_a, self.lam_b
                )

        ic(self.ks_rho, self.N_a, self.N_b, self.N)
        ic(self.lam)
        ic(self.eigs)
        ic(self.eta)
        ic(self.eta / np.sum(self.eta))
        return np.sqrt(self.eigs)

    def solve_buckling(
        self,
        x,
        ks_rho=10000.0,
        sigma=10.0,
        nodal_sols=None,
        nodal_vecs=None,
        solve=True,
        Qcrit=None,
        iter=0,
        iter_crit=0,
        tracking=False,
    ):
        if self.N > len(self.reduced):
            self.N = len(self.reduced)

        # Compute the density at each node
        self.rho = self.fltr.apply(x)

        # Compute detJ, Be, and Te
        gauss_pts = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
        xe = self.X[self.conn, 0]
        ye = self.X[self.conn, 1]

        self.detJ = np.zeros((2, 2, self.nelems))
        self.Be = np.zeros((2, 2, self.nelems, 3, 8))
        self.Be00 = np.zeros((self.nelems, 3, 8))
        self.Te = np.zeros((2, 2, self.nelems, 3, 4, 4))

        for j in range(2):
            for i in range(2):
                xi = gauss_pts[i]
                eta = gauss_pts[j]
                if not self.kokkos or self.check_kokkos:
                    Be_python = np.zeros((self.nelems, 3, 8))
                    Te_python = np.zeros((self.nelems, 3, 4, 4))
                    detJ_python = _populate_Be_and_Te(
                        self.nelems, xi, eta, xe, ye, Be_python, Te_python
                    )
                    if i == 0 and j == 0:
                        _populate_Be(self.nelems, 0.0, 0.0, xe, ye, self.Be00)

                if self.kokkos or self.check_kokkos:
                    import kokkos as kokkos

                    detJ_kokkos, Be_kokkos, Te_kokkos = kokkos.populate_Be_Te(
                        xi, eta, xe, ye
                    )
                    Be_kokkos = Be_kokkos.reshape((self.nelems, 3, 8))
                    Te_kokkos = Te_kokkos.reshape((self.nelems, 3, 4, 4))
                    if i == 0 and j == 0:
                        _, self.Be00 = kokkos.populate_Be(0.0, 0.0, xe, ye)
                        self.Be00 = self.Be00.reshape((self.nelems, 3, 8))

                if self.check_kokkos:
                    ic(np.allclose(detJ_python, detJ_kokkos))
                    ic(np.allclose(Be_python, Be_kokkos))
                    ic(np.allclose(Te_python, Te_kokkos))

                if not self.kokkos:
                    self.detJ[i, j, :] = detJ_python
                    self.Be[i, j, :, :, :] = Be_python
                    self.Te[i, j, :, :, :, :] = Te_python
                else:
                    self.detJ[i, j, :] = detJ_kokkos
                    self.Be[i, j, :, :, :] = Be_kokkos
                    self.Te[i, j, :, :, :, :] = Te_kokkos

        K = self.assemble_stiffness_matrix(self.rho)
        Kr = self.reduce_matrix(K)

        # Compute the solution path
        fr = self.reduce_vector(self.f)
        Kfact = linalg.factorized(Kr)
        ur = Kfact(fr)
        u = self.full_vector(ur)

        # Find the gemoetric stiffness matrix
        G = self.assemble_stress_stiffness(self.rho, u)
        Gr = self.reduce_matrix(G)

        # self.store_EG(Gr, Kr, ks_rho, self.N_a, self.N_b)
        # exit()

        # Find the eigenvalues closest to zero. This uses a shift and
        # invert strategy around sigma = 0, which means that the largest
        # magnitude values are closest to zero.
        if solve:
            print("Solving eigenvalue problem")
            start = time.time()

            if self.N == len(self.reduced) or (
                self.check_gradient and self.rho.dtype == np.complex128
            ):
                mu, Qr = eigh(Gr.todense(), Kr.todense())
                mu, Qr = mu[: self.N], Qr[:, : self.N]
                pass
            else:
                # Method 0: this is the fast and most accurate way to solve the eigenvalue problem
                try:
                    mu, Qr = sparse.linalg.eigsh(
                        Gr,
                        M=Kr,
                        k=self.N,
                        sigma=sigma,
                        which="SM",
                        maxiter=1000,
                        tol=1e-15,
                    )
                except sparse.linalg.ArpackNoConvergence:
                    print("ArpackNoConvergence 1")
                    mu, Qr = sparse.linalg.eigsh(
                        Gr,
                        M=Kr,
                        k=self.N,
                        sigma=sigma,
                        which="SM",
                        maxiter=1000,
                        tol=1e-8,
                    )
                # X = np.eye(Kr.shape[0], self.N)
                # M = sparse.diags(1.0 / np.diag(Kr.todense()))
                # mu_lobpcg, Qr_lobpcg = sparse.linalg.lobpcg(Gr, X, B=Kr,  M=M, largest=False, tol=1e-8, maxiter=1000)

            BLF = -1.0 / mu
            end = time.time()
            print("Solving eigenvalue problem: %f" % (end - start))
            # BLF = - (Q^T * K * Q) / (Q^T * G * Q) or (K + BLF * G) * Q = 0
            ic(np.allclose(Kr @ Qr + Gr @ Qr @ np.diag(BLF), 0.0, atol=1e-6))
        else:
            BLF = np.zeros(self.N)
            Qr = np.zeros((Kr.shape[0], self.N))
            mu = np.zeros(self.N)

        # self.QGQ = np.zeros(self.N, dtype=x.dtype)
        # for i in range(self.N):
        #     self.QGQ[i] = np.dot(Qr[:, i], Gr @ Qr[:, i])

        Q = np.zeros((self.nvars, self.N), dtype=x.dtype)
        for i in range(self.N):
            Q[self.reduced, i] = Qr[:, i]

        # Save vtk output data
        if nodal_sols is not None:
            nodal_sols["x"] = np.array(x)
            nodal_sols["rho"] = np.array(self.rho)

        if nodal_vecs is not None:
            for i in range(self.N):
                nodal_vecs["phi%d" % i] = [Q[0::2, i], Q[1::2, i]]

        # check if frequency is on the constraint

        self.A, self.B = G, K
        self.Kr, self.u, self.Bfact = Kr, u, Kfact

        self.dA = lambda q1, q2: self.stress_stiffness_derivative(
            self.rho, self.u, q1, q2, self.Bfact
        )
        self.dB = lambda q1, q2: self.stiffness_matrix_derivative(self.rho, q1, q2)

        self.eigs, self.lam, self.Q = BLF, mu, Q  # Ar @ Qr = lam * Br @ Qr

        self.D_index_modify = self.D_index

        # check the difference between the first 4 eigenvalues and find the index of the closest pair
        norm = np.zeros(6)
        index = 0
        min_diff = 1e10
        if tracking:
            for i in range(4):
                for j in range(i + 1, 4):
                    norm[index] = np.abs(self.lam[i] - self.lam[j])
                    if norm[index] < min_diff:
                        min_diff = norm[index]
                        self.N_a = i
                        self.N_b = j
                    index += 1

            ic(norm)
            ic(self.N_a, self.N_b)

        if len(self.D_index) != 1 and len(self.D_index) != 0:
            q_max = np.zeros(self.N_b - self.N_a + 1)
            D_indx_modify = np.zeros(self.N_b - self.N_a + 1)
            for i in range(self.N_a, self.N_b + 1):
                q = self.Q[self.D_index, i]
                q_max[i - self.N_a] = np.max(q)
                D_indx_modify[i - self.N_a] = self.D_index[np.argmax(q)]
                print(
                    "D_indx: %d, %d, %f,"
                    % (i, D_indx_modify[i - self.N_a], q_max[i - self.N_a])
                )
            self.D_index_modify = D_indx_modify.astype(int)

        if self.N_a == self.N_b:
            scale = 1e-6
        else:
            scale = 0.5

        self.lam_a = (
            self.lam[self.N_a]
            - np.abs(self.lam[self.N_a] - self.lam[np.min([self.N_a - 1, 0])]) * scale
        )
        self.lam_b = (
            self.lam[self.N_b]
            + np.abs(self.lam[self.N_b] - self.lam[np.min([self.N_b + 1, self.N - 1])])
            * scale
        )

        if self.atype:
            self.ks_rho = ks_rho
            self.eta = np.exp(-self.ks_rho * (self.lam - np.min(self.lam)))
            self.N_a = 0
            self.N_b = self.N - 1
        else:
            if self.fun == mp.exp:
                self.ks_rho = -ks_rho
                self.eta = -self.softmax_ab(
                    self.fun, self.ks_rho, self.lam, self.lam_a, self.lam_b
                )
            else:
                self.ks_rho = ks_rho
                self.eta = self.softmax_ab(
                    self.fun, self.ks_rho, self.lam, self.lam_a, self.lam_b
                )

        ic(self.ks_rho, self.N_a, self.N_b, self.N)
        ic(self.lam)
        ic(self.eigs)
        ic(self.eta / np.sum(self.eta))

        # self.store_EG(Gr, Kr, ks_rho, self.N_a, self.N_b)

        return self.eigs

    def store_EG(self, Gr, Kr, ks_rho, N_a, N_b):
        # if this function is called, no need to call it again
        if not hasattr(self, "called"):
            self.called = True
        else:
            return
        print("Storing E and G matrices")
        n = 100
        N_a = 10
        N_b = 20
        # mu = eigvalsh(Gr.todense(), Kr.todense(), eigvals=(0, n))
        mu, _ = sparse.linalg.eigsh(
            Gr,
            M=Kr,
            k=n,
            sigma=10.0,
            which="SM",
            maxiter=1000,
            tol=1e-15,
        )
        BLF = -1.0 / mu
        # ks_rho = 1000 * BLF[0]
        ks_rho = 1000.0
        ic(ks_rho)

        lam_a = BLF[N_a] - np.min(np.abs(BLF)) * 0.001
        lam_b = BLF[N_b] + np.min(np.abs(BLF)) * 0.001

        eta = self.softmax_ab(self.fun, ks_rho, BLF, lam_a, lam_b)
        eta_sum = np.sum(eta)

        ic(mu[:20])
        ic(eta[:20] / eta_sum)

        E = np.zeros((n, n))
        G = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                E[i, j] = self.Eij_ab(
                    self.fun, ks_rho, eta_sum, BLF[i], BLF[j], lam_a, lam_b
                )
                G[i, j] = self.Gij_ab(
                    self.fun, ks_rho, eta_sum, BLF[i], BLF[j], lam_a, lam_b
                )
        np.save("data/mu.npy", mu)
        np.save("data/E.npy", E)
        np.save("data/G.npy", G)

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

        omega = np.sqrt(self.eigs)

        c = np.min(omega)
        eta = np.exp(-ks_rho * (omega - c))
        a = np.sum(eta)
        eta = eta / a

        dfdrho = np.zeros(self.nnodes)

        for i in range(len(self.eigs)):
            kx = self.stiffness_matrix_derivative(self.rho, self.Q[:, i], self.Q[:, i])
            dfdrho += (eta[i] / (2 * omega[i])) * kx

            mx = self.mass_matrix_derivative(self.rho, self.Q[:, i], self.Q[:, i])
            dfdrho -= (omega[i] ** 2 * eta[i] / (2 * omega[i])) * mx

        return self.fltr.applyGradient(dfdrho, x)

    def ks_buckling(self, ks_rho=100.0):
        """
        Compute the smoothed approximation of the smallest eigenvalue
        """
        # c = np.min(self.eigs)
        # eta = np.exp(-ks_rho * (self.eigs - c))
        # a = np.sum(eta)
        # ks_min = c - np.log(a) / ks_rho

        mu = 1 / self.eigs
        c = np.max(mu)
        eta = np.exp(ks_rho * (mu - c))
        ks_min = c + np.log(np.sum(eta)) / ks_rho
        return ks_min

    def ks_buckling_derivative(self, x, ks_rho=100.0):
        # c = np.min(self.eigs)
        # eta = np.exp(-ks_rho * (self.eigs - c))
        # a = np.sum(eta)
        # eta *= 1.0 / a

        mu = 1 / self.eigs
        c = np.max(mu)
        eta = np.exp(ks_rho * (mu - c))
        eta *= 1.0 / np.sum(eta)

        dfdrho = np.zeros(self.nnodes)
        Q = self.Q

        for i in range(len(self.eigs)):
            kx = self.stiffness_matrix_derivative(self.rho, Q[:, i], Q[:, i])
            gx = self.stress_stiffness_derivative(
                self.rho, self.u, Q[:, i], Q[:, i], self.Bfact
            )
            # dfdrho -= eta[i] * (kx + self.eigs[i] * gx) / self.QGQ[i]
            dfdrho -= eta[i] * (gx + mu[i] * kx)  # / self.QGQ[i]

        return self.fltr.applyGradient(dfdrho, x)

    @time_this
    def eigenvector_displacement(self):
        sum_eta = np.sum(self.eta)
        eta = self.eta / sum_eta
        h = 0.0
        # for i in range(np.max([0, self.N_a - 1]), np.min([self.N_b + 2, self.N])):
        for i in range(len(eta)):
            q = self.Q[self.D_index_modify, i]
            h += eta[i] * np.dot(q, q)
        return h

    def eigenvector_displacement_magnitude(self):
        sum_eta = np.sum(self.eta)
        eta = self.eta / sum_eta

        h = np.zeros(self.Q.shape[0] // 2)
        for i in range(len(eta)):
            h += eta[i] * (self.Q[0::2, i] ** 2 + self.Q[1::2, i] ** 2)

        return h

    def KSmax(self, q, ks_rho):
        c = np.max(q)
        eta = np.exp(ks_rho * (q - c))
        ks_max = c + np.log(np.sum(eta)) / ks_rho
        return ks_max

    def eigenvector_displacement_max(self, ks_rho_dis_max=1.0):
        h = self.eigenvector_displacement_magnitude()
        ks_max = self.KSmax(h, ks_rho_dis_max)
        return ks_max

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

    def preciseG(self, rho, trace, lam_min, lam1, lam2):
        with mp.workdps(80):
            if lam1 == lam2:
                val = (
                    -rho * lam1 * mp.exp(-rho * (lam1 - lam_min)) / trace
                    + mp.exp(-rho * (lam1 - lam_min)) / trace
                )
            else:
                val = (
                    (
                        lam1 * mp.exp(-rho * (lam1 - lam_min))
                        - lam2 * mp.exp(-rho * (lam2 - lam_min))
                    )
                    / (mp.mpf(lam1) - mp.mpf(lam2))
                    / mp.mpf(trace)
                )
        return np.float64(val)

    def Eij_ab(self, fun, rho, trace, lam1, lam2, lam_a, lam_b):
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

    def Gij_ab(self, fun, rho, trace, lam1, lam2, lam_a, lam_b):
        with mp.workdps(80):
            a1 = fun(rho * (mp.mpf(lam1) - mp.mpf(lam_a)))
            b1 = fun(rho * (mp.mpf(lam1) - mp.mpf(lam_b)))
            a2 = fun(rho * (mp.mpf(lam2) - mp.mpf(lam_a)))
            b2 = fun(rho * (mp.mpf(lam2) - mp.mpf(lam_b)))

            eta1 = a1 - b1
            eta2 = a2 - b2

            if lam1 == lam2:
                val = -rho * lam1 * eta1 * (a1 + b1) / mp.mpf(trace) + eta1 / mp.mpf(
                    trace
                )
            else:
                val = (
                    (lam1 * eta1 - lam2 * eta2)
                    / (mp.mpf(lam1) - mp.mpf(lam2))
                    / mp.mpf(trace)
                )
        return np.float64(val)

    @time_this
    def eigenvector_displacement_deriv(self, x):
        """
        Approximately compute the forward derivative
        """

        def D(q):
            Dq = np.zeros(self.Q.shape[0])
            Dq[self.D_index_modify] = q[self.D_index_modify]
            return Dq

        # Compute the h values
        h = self.eigenvector_displacement()

        # Set the value of the derivative
        dhdrho = self.kernel(
            self.ks_rho,
            self.eta,
            self.lam,
            self.Q,
            h,
            self.A,
            self.B,
            self.dA,
            self.dB,
            D,
            self.reduced,
            self.fun,
            self.N_a,
            self.N_b,
            self.lam_a,
            self.lam_b,
        )

        return self.fltr.applyGradient(dhdrho, x)

    @time_this
    def eigenvector_displacement_max_deriv(self, x, ks_rho_dis_max=1.0):
        """
        Approximately compute the forward derivative
        """

        # Compute the h values
        h_magnitude = self.eigenvector_displacement_magnitude()

        # Compute the eta-h values
        eta_h = np.exp(ks_rho_dis_max * (h_magnitude - np.max(h_magnitude)))
        eta_h = eta_h / np.sum(eta_h)

        # compute the eta_h * h
        h = np.dot(h_magnitude, eta_h)

        # eta_h * Dq
        def D(q):
            eta_Dq = np.zeros(len(q))
            for i in range(len(eta_h)):
                eta_Dq[2 * i] = eta_h[i] * q[2 * i]
                eta_Dq[2 * i + 1] = eta_h[i] * q[2 * i + 1]
            return eta_Dq

        # Set the value of the derivative
        dhdrho = self.kernel(
            self.ks_rho,
            self.eta,
            self.lam,
            self.Q,
            h,
            self.A,
            self.B,
            self.dA,
            self.dB,
            D,
            self.reduced,
            self.fun,
            self.N_a,
            self.N_b,
            self.lam_a,
            self.lam_b,
        )

        return self.fltr.applyGradient(dhdrho, x)

    @time_this
    def postprocess_strain_stress(self, x, u, allowable=1.0):
        """
        Compute strain field and Von-mises stress given a displacement field
        """

        # Average the density to get the element-wise density
        rhoE = 0.25 * (
            self.rho[self.conn[:, 0]]
            + self.rho[self.conn[:, 1]]
            + self.rho[self.conn[:, 2]]
            + self.rho[self.conn[:, 3]]
        )

        # Compute the stress relaxation factor
        relax = rhoE / (rhoE + self.epsilon * (1.0 - rhoE))

        # Be = np.zeros((self.nelems, 3, 8))

        # # Compute the x and y coordinates of each element
        # xe = self.X[self.conn, 0]
        # ye = self.X[self.conn, 1]

        # # Compute the stress in the middle of the element
        # xi = 0.0
        # eta = 0.0
        # _populate_Be(self.nelems, xi, eta, xe, ye, Be)

        qe = np.zeros((self.nelems, 8))
        qe[:, ::2] = u[2 * self.conn]
        qe[:, 1::2] = u[2 * self.conn + 1]

        # Compute the stresses in each element
        strain = np.einsum("nik,nk -> ni", self.Be00, qe)
        s = np.einsum("ij,njk,nk -> ni", self.C0, self.Be00, qe)

        # Compute the von Mises stress
        vonmises = (
            relax
            * (s[:, 0] ** 2 + s[:, 1] ** 2 - s[:, 0] * s[:, 1] + 3.0 * s[:, 2] ** 2)
        ) / allowable**2

        return strain, np.sqrt(vonmises)

    @time_this
    def get_stress_values(self, rho, eta, Q, N_a, N_b, allowable=1.0):
        """
        Compute the stress at each element
        """
        sum_eta = np.sum(eta)
        eta = eta / sum_eta
        # Loop over all the eigenvalues
        # stress = np.zeros(self.nelems)
        # set stress maybe complex
        stress = np.zeros(self.nelems, dtype=rho.dtype)

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
                qe = np.zeros((self.nelems, 8), dtype=rho.dtype)
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
            # Be = np.zeros((self.nelems, 3, 8))

            # # Compute the x and y coordinates of each element
            # xe = self.X[self.conn, 0]
            # ye = self.X[self.conn, 1]

            # # Compute the stress in the middle of the element
            # xi = 0.0
            # eta_ = 0.0
            # _populate_Be(self.nelems, xi, eta_, xe, ye, Be)

            for k in range(len(eta)):
                qe = np.zeros((self.nelems, 8), dtype=rho.dtype)
                qe[:, ::2] = Q[2 * self.conn, k]
                qe[:, 1::2] = Q[2 * self.conn + 1, k]

                # Compute the stresses in each element
                s = np.einsum("ij,njk,nk -> ni", self.C0, self.Be00, qe)

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
    def eigenvector_stress(self, x, ks_rho_stress=10.0, allowable=1.0, cell_sols=None):
        # Compute the stress values
        stress = self.get_stress_values(
            self.rho, self.eta, self.Q, self.N_a, self.N_b, allowable=allowable
        )

        if cell_sols is not None:
            cell_sols["eigenvector_stress"] = stress

        # Now aggregate over the stress
        max_stress = np.max(stress)
        h = (
            max_stress
            + np.log(np.sum(np.exp(ks_rho_stress * (stress - max_stress))))
            / ks_rho_stress
        )

        return h

    @time_this
    def eigenvector_stress_derivative(self, x, ks_rho_stress=10.0, allowable=1.0):
        # Compute the stress values
        stress = self.get_stress_values(
            self.rho, self.eta, self.Q, self.N_a, self.N_b, allowable=allowable
        )

        # Now aggregate over the stress
        max_stress = np.max(stress)
        eta_stress = np.exp(ks_rho_stress * (stress - max_stress))
        eta_stress = eta_stress / np.sum(eta_stress)

        # if self.check_gradient:
        #     ic(stress)
        #     ic(eta_stress)

        def D(q):
            return self.get_stress_product(self.rho, eta_stress, q, allowable=allowable)

        h_eta = np.dot(eta_stress, stress)

        # Set the value of the derivative
        dhdrho = self.kernel(
            self.ks_rho,
            self.eta,
            self.lam,
            self.Q,
            h_eta,
            self.A,
            self.B,
            self.dA,
            self.dB,
            D,
            self.reduced,
            self.fun,
            self.N_a,
            self.N_b,
            self.lam_a,
            self.lam_b,
        )

        dhdrho += self.get_stress_values_deriv(
            self.rho,
            eta_stress,
            self.eta,
            self.Q,
            self.N_a,
            self.N_b,
            allowable=allowable,
        )

        return self.fltr.applyGradient(dhdrho, x)

    @time_this
    def get_stress_values_deriv(self, rho, eta_stress, eta, Q, N_a, N_b, allowable=1.0):
        dfdrhoE = np.zeros(self.nelems)
        sum_eta = np.sum(eta)
        eta = eta / sum_eta
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
            # Be = np.zeros((self.nelems, 3, 8))

            # # Compute the x and y coordinates of each element
            # xe = self.X[self.conn, 0]
            # ye = self.X[self.conn, 1]

            # # Compute the stress in the middle of the element
            # xi = 0.0
            # eta_ = 0.0
            # _populate_Be(self.nelems, xi, eta_, xe, ye, Be)

            for k in range(len(eta)):
                qe = np.zeros((self.nelems, 8))
                qe[:, ::2] = Q[2 * self.conn, k]
                qe[:, 1::2] = Q[2 * self.conn + 1, k]

                # Compute the stresses in each element
                s = np.einsum("ij,njk,nk -> ni", self.C0, self.Be00, qe)

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
            # Be = np.zeros((self.nelems, 3, 8))

            # # Compute the x and y coordinates of each element
            # xe = self.X[self.conn, 0]
            # ye = self.X[self.conn, 1]

            # # Compute the stress in the middle of the element
            # xi = 0.0
            # eta_ = 0.0
            # _populate_Be(self.nelems, xi, eta_, xe, ye, Be)

            qe = np.zeros((self.nelems, 8))
            qe[:, ::2] = q[2 * self.conn]
            qe[:, 1::2] = q[2 * self.conn + 1]

            # Compute the stresses in each element
            s = np.einsum("ij,njk,nk -> ni", self.C0, self.Be00, qe)

            ds = np.zeros((self.nelems, 3))
            ds[:, 0] = eta_stress * relax * (s[:, 0] - 0.5 * s[:, 1]) / allowable**2
            ds[:, 1] = eta_stress * relax * (s[:, 1] - 0.5 * s[:, 0]) / allowable**2
            ds[:, 2] = eta_stress * relax * 3.0 * s[:, 2] / allowable**2

            Dqe = np.einsum("ni,ij,njk -> nk", ds, self.C0, self.Be00)

        Dq = np.zeros(self.nvars)
        for i in range(4):
            np.add.at(Dq, 2 * self.conn[:, i], Dqe[:, 2 * i])
            np.add.at(Dq, 2 * self.conn[:, i] + 1, Dqe[:, 2 * i + 1])

        return Dq

    @time_this
    def kernel(
        self,
        ks_rho,
        eta,
        lam,
        Q,
        h,
        A,
        B,
        dA,
        dB,
        D,
        reduced,
        fun,
        N_a,
        N_b,
        lam_a,
        lam_b,
    ):
        """
        For displacement constraints:
            a = None
            b = sum_i^N sum_j^N (q_i^T * D * q_j - δ_{ij} * h)
        For stress constraints:
            a = sum_i^N sum_k^nelems eta_stress_k * q_i^T * dD_k * q_i
            b = sum_i^N sum_k^nelems eta_stress_k * q_i^T * D_k * q_j - δ_{ij} * eta_stress_k * h_k

        Dq = D * q
        dA = dA(u, v)
        dB = dB(u, v)
        """

        Ar = self.reduce_matrix(A)
        Br = self.reduce_matrix(B)

        if self.check_gradient:
            Qr = self.reduce_vector(Q)
            ic(np.allclose(Ar @ Qr, Br @ Qr @ np.diag(lam)))

        dh = np.zeros(self.nnodes)
        eta_sum = np.sum(eta)
        eta = eta / eta_sum

        for j in range(np.max([0, N_a - 1]), np.min([N_b + 2, self.N])):
            # for j in range(self.N):
            for i in range(j + 1):
                qDq = Q[:, i].T @ D(Q[:, j])
                qAdotq = dA(Q[:, i], Q[:, j])
                qBdotq = dB(Q[:, i], Q[:, j])

                if i == j:
                    scalar = qDq - h
                else:
                    scalar = 2 * qDq

                if self.atype:
                    Eij = self.precise(ks_rho, eta_sum, np.min(lam), lam[i], lam[j])
                    Gij = self.preciseG(ks_rho, eta_sum, np.min(lam), lam[i], lam[j])
                else:
                    Eij = self.Eij_ab(
                        fun, ks_rho, eta_sum, lam[i], lam[j], lam_a, lam_b
                    )
                    Gij = self.Gij_ab(
                        fun, ks_rho, eta_sum, lam[i], lam[j], lam_a, lam_b
                    )

                dh += scalar * (Eij * qAdotq - Gij * qBdotq)

        Br = Br.tocsc()
        C = B.dot(Q[:, : np.min([N_b + 2, self.N])])
        # C = B.dot(Q[:, : self.N])
        nr = len(reduced)
        Cr = np.zeros((nr, C.shape[1]))
        for k in range(C.shape[1]):
            Cr[:, k] = C[reduced, k]
        Ur, R = np.linalg.qr(Cr)

        # Form a full factorization for the preconditioner
        factor = 0.99  # Should always be < 1 to ensure P is positive definite.
        P = Ar - factor * lam[0] * Br
        P = P.tocsc()
        Pfactor = linalg.factorized(P)

        def preconditioner(x):
            y = Pfactor(x)
            t = np.dot(Ur.T, y)
            y = y - np.dot(Ur, t)
            return y

        preop = linalg.LinearOperator((nr, nr), preconditioner)

        # Form the augmented linear system of equations
        for k in range(np.max([0, N_a - 1]), np.min([N_b + 2, self.N])):
            # for k in range(self.N):
            # Compute B * uk = D * qk
            Dq = D(Q[:, k])
            bkr = -2 * eta[k] * Dq[reduced]

            # Form the matrix
            def matrix(x):
                y = Ar.dot(x) - lam[k] * Br.dot(x)
                t = np.dot(Ur.T, y)
                y = y - np.dot(Ur, t)
                return y

            matop = linalg.LinearOperator((nr, nr), matrix)

            # Solve the augmented system of equations for vk
            t = np.dot(Ur.T, bkr)
            bkr = bkr - np.dot(Ur, t)
            time_start = time.time()
            phir, state = linalg.gmres(
                matop, bkr, M=preop, atol=1e-8, tol=1e-8, maxiter=100
            )  # atol=1e-15, tol=1e-10
            time_end = time.time()
            print(
                "GMRES %d/%d, state %d, time %f"
                % (k + 1, self.N, state, time_end - time_start)
            )
            phi = self.full_vector(phir)
            dh += dA(Q[:, k], phi) - lam[k] * dB(Q[:, k], phi)

            # qDq = np.dot(Q[:, k], Dq)
            # dh -= eta[k] * qDq * dB(Q[:, k], Q[:, k])
            dh -= eta[k] * h * dB(Q[:, k], Q[:, k])

        return dh

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
        dis_ub_frac=None,
        ks_rho_natural_freq=100,
        ks_rho_buckling=10000,
        ks_rho_stress=10,
        ks_rho_freq=100,
        ks_rho_dis_max=1.0,
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
        BLF_lb=None,
        stress_ub=None,
        compliance_ub=None,
        frequency_scale=1e-2,
        sigma_fixed=True,
        sigma_scale=1,
        stress_scale=1e-12,
        compliance_scale=1e6,
        weight=0.5,
        c0=1e-6,
        mu_ks0=1.0,
        check_gradient=False,
        domain="square",
        iter_crit=0,
        iter_crit_dis=0,
        restart_beta_p=True,
        iter_crit_stress=0,
        iter_crit_w=0,
        iter_crit_d=0,
        delta_beta=0.1,
        delta_p=0.01,
        tracking=False,
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
        self.BLF_lb = BLF_lb
        self.stress_ub = stress_ub
        self.compliance_ub = compliance_ub
        self.frequency_scale = frequency_scale
        self.sigma_fixed = sigma_fixed
        self.sigma_scale = sigma_scale
        self.stress_scale = stress_scale
        self.compliance_scale = compliance_scale
        self.weight = weight
        self.c0 = c0
        self.mu_ks0 = mu_ks0
        self.lb = lb
        self.check_gradient = check_gradient
        self.domain = domain
        self.iter_crit = iter_crit
        self.iter_crit_dis = iter_crit_dis
        self.iter_crit_stress = iter_crit_stress
        self.iter_crit_w = iter_crit_w
        self.iter_crit_d = iter_crit_d
        self.restart_beta_p = restart_beta_p
        self.delta_beta = delta_beta
        self.delta_p = delta_p
        self.tracking = tracking

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
        self.fixed_dis_ub_frac = dis_ub_frac

        self.ks_rho_natural_freq = ks_rho_natural_freq
        self.ks_rho_buckling = ks_rho_buckling
        self.ks_rho_stress = ks_rho_stress
        self.ks_rho_freq = ks_rho_freq
        self.ks_rho_dis_max = ks_rho_dis_max

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
        self.Qcrit = None
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
        if self.check_gradient:
            np.random.seed(0)
            x[:] = 0.5 + 0.5 * np.random.uniform(size=len(x))
        return

    def evalObjCon(self, x, eval_all=False):
        # Functions of interest to be logged
        foi = OrderedDict()
        foi["area"] = "n/a"
        if self.prob == "natural_frequency":
            foi["omega_ks"] = "n/a"
        elif self.prob == "buckling":
            foi["BLF_ks"] = "n/a"
            if self.objf == "compliance-buckling":
                foi["compliance"] = "n/a"
        foi["stress_ks"] = "n/a"

        t_start = timer()

        vtk_nodal_sols = None
        vtk_nodal_vecs = None
        vtk_cell_sols = None
        vtk_cell_vecs = None
        vtk_path = None
        stress_ks = None

        print("")
        print("it_counter: ", self.it_counter)

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
        # compute ks_rho, N, and eta
        if self.objf == "compliance" and self.confs == ["volume_ub"]:
            solve = False
        else:
            solve = True

        if self.it_counter == 0:
            sigma = 100
        elif self.sigma_fixed == False:
            sigma = self.analysis.eigs[0] * self.sigma_scale
        else:
            sigma = self.sigma_scale
        sigma = np.min([sigma, 1e3])
        sigma = np.max([sigma, 1e-2])
        ic(sigma)

        if self.prob == "natural_frequency":
            omega = self.analysis.solve_eigenvalue_problem(
                self.xfull,
                ks_rho=self.ks_rho_natural_freq,
                nodal_sols=vtk_nodal_sols,
                nodal_vecs=vtk_nodal_vecs,
                solve=solve,
            )
        elif self.prob == "buckling":
            if self.tracking:
                if self.it_counter == self.iter_crit_dis and self.iter_crit_dis != 0:
                    self.Qcrit = self.analysis.Q[:, self.analysis.N_a]

            BLF = self.analysis.solve_buckling(
                self.xfull,
                ks_rho=self.ks_rho_buckling,
                sigma=sigma,
                nodal_sols=vtk_nodal_sols,
                nodal_vecs=vtk_nodal_vecs,
                solve=solve,
                Qcrit=self.Qcrit,
                iter=self.it_counter,
                iter_crit=self.iter_crit_dis,
                tracking=self.tracking,
            )

        # Evaluate objectives
        if self.objf == "volume":
            area = self.analysis.eval_area(self.xfull)
            obj = area
            if (
                self.domain == "beam"
                or self.domain == "building"
                or self.domain == "leg"
                or self.domain == "rhombus"
            ):
                area *= 1 / np.sum(self.area_gradient)
            foi["area"] = area
        elif self.objf == "frequency":
            if self.prob == "natural_frequency":
                omega_ks = self.analysis.ks_omega(ks_rho=self.ks_rho_freq)
                obj = -self.frequency_scale * omega_ks
                foi["omega_ks"] = omega_ks
            elif self.prob == "buckling":
                mu_ks = self.analysis.ks_buckling(ks_rho=self.ks_rho_freq)
                obj = self.frequency_scale * mu_ks
                foi["BLF_ks"] = 1 / mu_ks
        elif self.objf == "compliance":
            compliance = self.analysis.compliance(self.xfull)
            obj = self.compliance_scale * compliance
            foi["compliance"] = compliance
        elif self.objf == "compliance-buckling":
            if self.prob == "natural_frequency":
                print("compliance-buckling is not available for natural frequency")
                exit()

            compliance = self.analysis.compliance(self.xfull)
            mu_ks = self.analysis.ks_buckling(ks_rho=self.ks_rho_freq)
            obj = self.weight * (compliance / self.c0) + (1 - self.weight) * (
                mu_ks / self.mu_ks0
            )

            foi["compliance"] = compliance
            foi["BLF_ks"] = 1 / mu_ks

            ic((compliance / self.c0))
            ic((mu_ks / self.mu_ks0))

        elif self.objf == "displacement":
            displacement = self.analysis.eigenvector_displacement()
            obj = 10 * displacement
            foi["displacement"] = displacement
        else:  # objf == "stress"
            stress_ks = self.analysis.eigenvector_stress(
                self.xfull, self.ks_rho_stress, cell_sols=vtk_cell_sols
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
            # if self.it_counter  can be devide by 10, and stress is not in the confs, then compute stress
            # if self.it_counter % 5 == 0 and "stress" not in self.confs:
            # stress_ks = self.analysis.eigenvector_stress(
            #     self.xfull, self.ks_rho_stress, cell_sols=vtk_cell_sols
            # )
            if self.tracking:
                foi["stress_ks"] = self.analysis.N_a + self.analysis.N_b * 0.1

        if "volume_lb" in self.confs:
            assert self.fixed_area_lb is not None
            area = self.analysis.eval_area(self.xfull)
            con.append(area - self.fixed_area_lb)
            if "volume_ub" not in self.confs:
                foi["area"] = area / np.sum(self.area_gradient)

        if "frequency" in self.confs:
            if self.prob == "natural_frequency":
                assert self.omega_lb is not None
                omega_ks = self.analysis.ks_omega(ks_rho=self.ks_rho_freq)
                con.append(omega_ks - self.omega_lb)
                foi["omega_ks"] = omega_ks
            elif self.prob == "buckling":
                assert self.BLF_lb is not None
                mu_ks = self.analysis.ks_buckling(ks_rho=self.ks_rho_freq)
                mu_ub = 1 / self.BLF_lb
                con.append(1 - mu_ks / mu_ub)
                foi["BLF_ks"] = 1 / mu_ks

        if "stress" in self.confs:
            assert self.stress_ub is not None
            stress_ks = self.analysis.eigenvector_stress(
                self.xfull, self.ks_rho_stress, cell_sols=vtk_cell_sols
            )
            if self.it_counter < self.iter_crit_stress:
                con.append(1.0)
            else:
                con.append(1.0 - stress_ks / self.stress_ub)
                print("stress added to constraints in iteration %d" % self.it_counter)
            foi["stress_ks"] = stress_ks

        if "displacement" in self.confs:
            assert self.fixed_dis_ub is not None
            displacement = self.analysis.eigenvector_displacement()
            if self.it_counter < self.iter_crit_dis:
                con.append(1.0)
            else:
                con.append(self.fixed_dis_ub - displacement)
            foi["displacement"] = displacement

        if "displacement_frac" in self.confs:
            assert self.fixed_dis_ub_frac is not None
            displacement = self.analysis.eigenvector_displacement()
            displacement_max = self.analysis.eigenvector_displacement_max(
                self.ks_rho_dis_max
            )
            con.append(self.fixed_dis_ub_frac * displacement_max - displacement)
            ic(displacement, displacement_max)
            foi["dis_frac"] = displacement / displacement_max

        if "compliance" in self.confs:
            assert self.compliance_ub is not None
            compliance = self.analysis.compliance(self.xfull)
            con.append(1.0 - compliance / self.compliance_ub)
            foi["compliance"] = compliance

        # Evaluate all quantities of interest
        if eval_all:
            stress_ks = self.analysis.eigenvector_stress(
                self.xfull, self.ks_rho_stress, cell_sols=vtk_cell_sols
            )
            if self.prob == "natural_frequency":
                omega_ks = self.analysis.ks_omega(ks_rho=self.ks_rho_freq)
                foi["omega_ks"] = omega_ks
            elif self.prob == "buckling":
                mu_ks = self.analysis.ks_buckling(ks_rho=self.ks_rho_freq)
                foi["BLF_ks"] = 1 / mu_ks
            foi["stress_ks"] = stress_ks

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
            utils.to_vtk(
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
                if self.prob == "natural_frequency":
                    for i in range(len(omega)):
                        name = "omega[%d]" % i
                        f.write("%25s" % name)
                elif self.prob == "buckling":
                    for i in range(len(BLF)):
                        name = "BLF[%d]" % i
                        f.write("%25s" % name)
                f.write("\n")

            f.write("%10d" % self.it_counter)
            if self.prob == "natural_frequency":
                for i in range(len(omega)):
                    f.write("%25.15e" % omega[i])
            elif self.prob == "buckling":
                for i in range(len(BLF)):
                    f.write("%25.15e" % BLF[i])
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

        # update the filter beta: after step 100, increase it by 1 every 25 steps, up to 16
        if self.it_counter > self.iter_crit:
            if (self.it_counter - self.iter_crit) % 1 == 0:
                self.analysis.fltr.beta += self.delta_beta
                self.analysis.fltr.beta = min(self.analysis.fltr.beta, 16)

        if self.it_counter > self.iter_crit:
            if (self.it_counter - self.iter_crit) % 1 == 0:
                self.analysis.p += self.delta_p
                self.analysis.p = min(self.analysis.p, 6)

        if self.restart_beta_p:
            if (
                self.it_counter == self.iter_crit_dis
                or self.it_counter == self.iter_crit_stress
            ):
                self.analysis.p = 3.0
                self.analysis.fltr.beta = 1e-6

        if self.iter_crit_w != 0:
            if self.it_counter % self.iter_crit_w == 0 and self.it_counter != 0:
                self.weight -= 0.1
                self.weight = min(self.weight, 1.0)
                self.weight = max(self.weight, 0.0)

        if self.iter_crit_d != 0:
            if self.it_counter % self.iter_crit_d == 0 and self.it_counter != 0:
                self.fixed_dis_ub -= 1.0
                self.fixed_dis_ub = max(self.fixed_dis_ub, 0.0)

        ic(self.weight, self.fixed_dis_ub)
        ic(self.analysis.fltr.beta, self.analysis.p)

        self.it_counter += 1
        return fail, obj, con

    def evalObjConGradient(self, x, g, A):
        t_start = timer()

        # Map x to the full design space if needed
        if self.dv_mapping is not None:
            self.xfull[:] = self.dv_mapping.dot(x)
            self.xfull[self.non_design_nodes] = 1.0
        else:
            self.xfull[self.design_nodes] = x[:]

        if self.objf == "volume":
            if self.dv_mapping is not None:
                g[:] = self.dv_mapping.T.dot(
                    self.analysis.eval_area_gradient(self.xfull)
                )
            else:
                g[:] = self.analysis.eval_area_gradient(self.xfull)[self.design_nodes]

        elif self.objf == "frequency":
            if self.prob == "natural_frequency":
                if self.dv_mapping is not None:
                    g[:] = -self.frequency_scale * self.dv_mapping.T.dot(
                        self.analysis.ks_omega_derivative(self.xfull, self.ks_rho_freq)
                    )
                else:
                    g[:] = (
                        -self.frequency_scale
                        * self.analysis.ks_omega_derivative(
                            self.xfull, self.ks_rho_freq
                        )[self.design_nodes]
                    )
            elif self.prob == "buckling":
                if self.dv_mapping is not None:
                    g[:] = self.frequency_scale * self.dv_mapping.T.dot(
                        self.analysis.ks_buckling_derivative(
                            self.xfull, self.ks_rho_freq
                        )
                    )
                else:
                    g[:] = (
                        self.frequency_scale
                        * self.analysis.ks_buckling_derivative(
                            self.xfull, self.ks_rho_freq
                        )[self.design_nodes]
                    )
        elif self.objf == "compliance":
            if self.dv_mapping is not None:
                g[:] = self.compliance_scale * self.dv_mapping.T.dot(
                    self.analysis.compliance_gradient(self.xfull)
                )
            else:
                g[:] = (
                    self.compliance_scale
                    * self.analysis.compliance_gradient(self.xfull)[self.design_nodes]
                )

        elif self.objf == "compliance-buckling":
            d_compliance = self.analysis.compliance_gradient(self.xfull)
            d_buckling = self.analysis.ks_buckling_derivative(
                self.xfull, self.ks_rho_freq
            )
            d_compliance_buckling = (
                self.weight * d_compliance / self.c0
                + (1 - self.weight) * d_buckling / self.mu_ks0
            )

            if self.dv_mapping is not None:
                g[:] = self.dv_mapping.T.dot(d_compliance_buckling)
            else:
                g[:] = d_compliance_buckling[self.design_nodes]

        elif self.objf == "displacement":
            if self.dv_mapping is not None:
                g[:] = 10 * self.dv_mapping.T.dot(
                    self.analysis.eigenvector_displacement_deriv(self.xfull)
                )
            else:
                g[:] = (
                    10
                    * self.analysis.eigenvector_displacement_deriv(self.xfull)[
                        self.design_nodes
                    ]
                )
        else:  # objf == "stress"
            if self.dv_mapping is not None:
                g[:] = self.stress_scale * self.dv_mapping.T.dot(
                    self.analysis.eigenvector_stress_derivative(
                        self.xfull, self.ks_rho_stress
                    )
                )
            else:
                g[:] = (
                    self.stress_scale
                    * self.analysis.eigenvector_stress_derivative(
                        self.xfull, self.ks_rho_stress
                    )[self.design_nodes]
                )

        # Evaluate constraint gradients
        index = 0

        if "volume_ub" in self.confs:
            if self.dv_mapping is not None:
                A[index][:] = -self.dv_mapping.T.dot(
                    self.analysis.eval_area_gradient(self.xfull)
                )
            else:
                A[index][:] = -self.analysis.eval_area_gradient(self.xfull)[
                    self.design_nodes
                ]
            index += 1

        if "volume_lb" in self.confs:
            if self.dv_mapping is not None:
                A[index][:] = self.dv_mapping.T.dot(
                    self.analysis.eval_area_gradient(self.xfull)
                )
            else:
                A[index][:] = self.analysis.eval_area_gradient(self.xfull)[
                    self.design_nodes
                ]
            index += 1

        if "frequency" in self.confs:
            if self.prob == "natural_frequency":
                if self.dv_mapping is not None:
                    A[index][:] = self.dv_mapping.T.dot(
                        self.analysis.ks_omega_derivative(self.xfull)
                    )
                else:
                    A[index][:] = self.analysis.ks_omega_derivative(self.xfull)[
                        self.design_nodes
                    ]
            elif self.prob == "buckling":
                if self.dv_mapping is not None:
                    A[index][:] = -self.dv_mapping.T.dot(
                        self.analysis.ks_buckling_derivative(self.xfull)
                    )
                else:
                    A[index][:] = -self.analysis.ks_buckling_derivative(self.xfull)[
                        self.design_nodes
                    ]
            index += 1

        if "stress" in self.confs:
            if self.dv_mapping is not None:
                if self.it_counter < self.iter_crit_stress:
                    A[index][:] = 0.0
                else:
                    A[index][:] = (
                        -self.dv_mapping.T.dot(
                            self.analysis.eigenvector_stress_derivative(
                                self.xfull, self.ks_rho_stress
                            )
                        )
                        / self.stress_ub
                    )
            else:
                if self.it_counter < self.iter_crit_stress:
                    A[index][:] = 0.0
                else:
                    A[index][:] = (
                        -self.analysis.eigenvector_stress_derivative(
                            self.xfull, self.ks_rho_stress
                        )[self.design_nodes]
                        / self.stress_ub
                    )
            index += 1

        if "displacement" in self.confs:
            if self.dv_mapping is not None:
                if self.it_counter < self.iter_crit_dis:
                    A[index][:] = 0.0
                else:
                    A[index][:] = -self.dv_mapping.T.dot(
                        self.analysis.eigenvector_displacement_deriv(self.xfull)
                    )
            else:
                if self.it_counter < self.iter_crit_dis:
                    A[index][:] = 0.0
                else:
                    A[index][:] = -self.analysis.eigenvector_displacement_deriv(
                        self.xfull
                    )[self.design_nodes]
            index += 1

        if "displacement_frac" in self.confs:
            dis_deriv = self.analysis.eigenvector_displacement_deriv(self.xfull)
            dis_max_deriv = self.analysis.eigenvector_displacement_max_deriv(
                self.xfull, self.ks_rho_dis_max
            )
            grad = self.fixed_dis_ub_frac * dis_max_deriv - dis_deriv
            if self.dv_mapping is not None:
                A[index][:] = self.dv_mapping.T.dot(grad)
            else:
                A[index][:] = grad[self.design_nodes]
            index += 1

        if "compliance" in self.confs:
            if self.dv_mapping is not None:
                A[index][:] = (
                    -self.dv_mapping.T.dot(
                        self.analysis.compliance_gradient(self.xfull)
                    )
                    / self.compliance_ub
                )
            else:
                A[index][:] = (
                    -self.analysis.compliance_gradient(self.xfull)[self.design_nodes]
                    / self.compliance_ub
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


def gradient_check(topo, problem):
    ks_rho_stress = 10.0

    def init(x):
        if problem == "natural_frequency":
            _ = topo.solve_eigenvalue_problem(x)

        elif problem == "buckling":
            _ = topo.solve_buckling(x)

    def compute_u(f, K):
        fr = topo.reduce_vector(f)
        Kr = topo.reduce_matrix(K)
        Kfact = linalg.factorized(Kr)
        ur = Kfact(fr)
        u = topo.full_vector(ur)
        return u, Kfact

    # use the complex step method to check the derivative
    dh = 1e-5
    x = 0.5 + 0.5 * np.random.uniform(size=topo.nnodes)
    px = np.random.RandomState(0).rand(topo.nnodes)
    q1 = np.random.RandomState(1).rand(topo.nvars)
    q2 = np.random.RandomState(2).rand(topo.nvars)

    init(x)
    # lam_a, lam_b = topo.lam_a, topo.lam_b
    K = topo.assemble_stiffness_matrix(topo.rho)
    u, Kfact = compute_u(topo.f, K)

    if problem == "natural_frequency":
        d_omega = topo.ks_omega_derivative(topo.rho) @ px
    elif problem == "buckling":
        d_omega = topo.ks_buckling_derivative(topo.rho) @ px

    dK = topo.stiffness_matrix_derivative(topo.rho, q1, q2) @ px
    dM = topo.mass_matrix_derivative(topo.rho, q1, q2) @ px
    dG = topo.stress_stiffness_derivative(topo.rho, u, q1, q2, Kfact) @ px
    dis_grad = topo.eigenvector_displacement_deriv(x) @ px
    stress_grad = topo.eigenvector_stress_derivative(x, ks_rho_stress) @ px
    dis_max_grad = topo.eigenvector_displacement_max_deriv(x) @ px
    c_grad = topo.compliance_gradient(x) @ px

    # s = topo.get_stress_values(rho, topo.eta, topo.Q)
    # eta_stress = np.exp(ks_rho_stress * (s - np.max(s)))
    # eta_stress = eta_stress / np.sum(eta_stress)
    # ds = topo.get_stress_values_deriv(rho, eta_stress, topo.eta, topo.Q) @ px

    # use central difference to check the derivative
    init(x + dh * px)
    # topo.lam_a, topo.lam_b = lam_a, lam_b
    if problem == "natural_frequency":
        omega1 = topo.ks_omega()
    elif problem == "buckling":
        omega1 = topo.ks_buckling()

    dis1 = topo.eigenvector_displacement()
    stress1 = topo.eigenvector_stress(x + dh * px, ks_rho_stress)
    dis_max1 = topo.eigenvector_displacement_max()
    c1 = topo.compliance(x + dh * px)
    # s1 = topo.get_stress_values(topo.rho, topo.eta, topo.Q)

    init(x - dh * px)
    # topo.lam_a, topo.lam_b = lam_a, lam_b
    if problem == "natural_frequency":
        omega2 = topo.ks_omega()
    elif problem == "buckling":
        omega2 = topo.ks_buckling()

    dis2 = topo.eigenvector_displacement()
    stress2 = topo.eigenvector_stress(x - dh * px, ks_rho_stress)
    dis_max2 = topo.eigenvector_displacement_max()
    c2 = topo.compliance(x - dh * px)
    # s2 = topo.get_stress_values(topo.rho, topo.eta, topo.Q)

    dis_grad_cf = (dis1 - dis2) / (2 * dh)
    stress_grad_cf = (stress1 - stress2) / (2 * dh)
    omega_grad_cf = (omega1 - omega2) / (2 * dh)
    dis_max_grad_cf = (dis_max1 - dis_max2) / (2 * dh)
    c_grad_cf = (c1 - c2) / (2 * dh)
    # s_grad_cf = (s1 - s2) / (2 * dh)
    ic(dis_grad_cf)
    ic(dis_grad)
    ic(dis_max_grad_cf)
    ic(dis_max_grad)
    ic(c_grad)
    ic(c_grad_cf)

    error_d_dis_cf = np.abs(dis_grad_cf - dis_grad) / np.abs(dis_grad_cf)
    error_d_stress_cf = np.abs(stress_grad_cf - stress_grad) / np.abs(stress_grad_cf)
    error_d_omega_cf = np.abs(omega_grad_cf - d_omega) / np.abs(omega_grad_cf)
    error_d_dis_max_cf = np.abs(dis_max_grad_cf - dis_max_grad) / np.abs(
        dis_max_grad_cf
    )
    error_d_c_cf = np.abs(c_grad_cf - c_grad) / np.abs(c_grad_cf)
    # error_s_cf = np.abs(s_grad_cf - ds) / np.abs(s_grad_cf)

    ic(error_d_dis_cf)
    ic(error_d_stress_cf)
    ic(error_d_omega_cf)
    ic(error_d_dis_max_cf)
    ic(error_d_c_cf)
    # ic(error_s_cf)

    # dh = 1e-15
    # init(x + dh * px * 1j)
    # dis1 = topo.eigenvector_displacement()
    # stress1 = topo.eigenvector_stress(x + dh * px * 1j, topo.ks_rho, ks_rho_stress)
    # K1 = topo.assemble_stiffness_matrix(topo.rho)
    # M1 = topo.assemble_mass_matrix(topo.rho)
    # u1, _ = compute_u(topo.f, K1)
    # G1 = topo.assemble_stress_stiffness(topo.rho, u1)

    # dis_grad_cs = np.imag(dis1) / dh
    # stress_grad_cs = np.imag(stress1) / dh
    # K_grad_cs = q1.T @ np.imag(K1) @ q2 / dh
    # M_grad_cs = q1.T @ np.imag(M1) @ q2 / dh
    # G_grad_cs = q1.T @ np.imag(G1) @ q2 / dh

    # ic(dis_grad_cf)
    # ic(dis_grad_cs)
    # ic(dis_grad)
    # ic(stress_grad_cf)
    # ic(stress_grad_cs)
    # ic(stress_grad)

    # error_dis_cs = np.abs(dis_grad_cs - dis_grad) / np.abs(dis_grad_cs)
    # error_stress_cs = np.abs(stress_grad_cs - stress_grad) / np.abs(stress_grad_cs)
    # error_K_cs = np.abs(K_grad_cs - dK) / np.abs(K_grad_cs)
    # error_M_cs = np.abs(M_grad_cs - dM) / np.abs(M_grad_cs)
    # error_G_cs = np.abs(G_grad_cs - dG) / np.abs(G_grad_cs)

    # ic(error_dis_cs)
    # ic(error_stress_cs)
    # ic(error_K_cs)
    # ic(error_M_cs)
    # ic(error_G_cs)


def main(args):
    args = utils.create_folder(args)
    # Set up logger
    Logger.set_log_path(os.path.join(args.prefix, "stdout.log"))
    timer_set_log_path(os.path.join(args.prefix, "profiler.log"))

    # Save option values
    with open(os.path.join(args.prefix, "options.txt"), "w") as f:
        f.write("Options:\n")
        for k, v in vars(args).items():
            f.write(f"{k:<20}{v}\n")

    if args.domain == "square":
        conn, X, r0, bcs, forces, non_design_nodes, dv_mapping = domain.square(
            r0_=args.r0, nx=args.nx, m0_block_frac=args.m0_block_frac
        )
    elif args.domain == "lbracket":
        conn, X, r0, bcs, forces, non_design_nodes = domain.lbracket(
            r0_=args.r0, nx=args.nx, m0_block_frac=args.m0_block_frac
        )
        dv_mapping = None
    elif args.domain == "beam":
        conn, X, r0, bcs, forces, non_design_nodes, dv_mapping = domain.beam(
            r0_=args.r0, nx=args.nx, prob=args.problem
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
        ) = domain.building(r0_=args.r0, nx=args.nx, m0_block_frac=args.m0_block_frac)
    elif args.domain == "leg":
        conn, X, r0, bcs, forces, non_design_nodes = domain.leg(r0_=args.r0, nx=args.nx)
        dv_mapping = None
    elif args.domain == "rhombus":
        conn, X, r0, bcs, forces, non_design_nodes = domain.rhombus(
            r0_=args.r0, nx=args.nx
        )
        dv_mapping = None

    # for there is displacement constraint, we need to use the displacement constraint
    if "beam" in args.domain:
        m = args.nx
        n = int(np.ceil((args.nx / 3)))
    elif "building" in args.domain or "leg" in args.domain:
        m = args.nx
        n = int(np.ceil((2 * args.nx)))
    elif "square" in args.domain:
        m = args.nx
        n = args.nx
    elif "rhombus" in args.domain:
        m = int(3 * args.nx)
        n = args.nx
    else:
        raise ValueError("Not supported domain")

    D_index = np.array([])
    indx = np.array([])
    ic(args.domain, args.objf, args.confs, m, n)
    for conf in args.confs:
        if conf == "displacement" or conf == "displacement_frac":
            if args.domain == "beam":
                if args.mode == 1:
                    # node_loc=(n, m * 0.5), y direction
                    # indx = int((m * 0.5))
                    indx = np.append(indx, int((n * m - m * 0.5)) - 1)
                    D_index = np.append(D_index, 2 * indx + 1)
                elif args.mode == 2:
                    # node_loc=(n, m * 0.5), y direction
                    D_index = np.append(D_index, 2 * (m // 2 - 1) + 1)
                elif args.mode == 3:
                    for i in range(m):
                        indx = np.append(indx, i)
                        indx = np.append(indx, n * m - 1 - i)
                        # D_index = np.append(D_index, 2 * i + 1)
                        D_index = np.append(D_index, 2 * (n * m - 1 - i) + 1)
                elif args.mode == 4:
                    for i in range(m):
                        indx = np.append(indx, n / 2 * m - 1 - i)
                        indx = np.append(indx, n / 2 * m + i)
                        D_index = np.append(D_index, 2 * (n / 2 * m - 1 - i) + 1)
                        D_index = np.append(D_index, 2 * (n / 2 * m + i) + 1)
            if args.domain == "square":
                if args.mode == 1:
                    # node_loc=(0.75*n, 0.75*m), x direction
                    # indx = int((0.1 * n * (m + 1) + 0.1 * m))
                    # node_loc=(0.5*n, 0.5*m)
                    indx = np.append(indx, int((0.34 * n * (m + 1) + 0.34 * m)) - 1)
                    D_index = np.append(D_index, 2 * indx)
                    D_index = np.append(D_index, 2 * indx + 1)
                elif args.mode == 2:
                    indx = np.append(indx, int((0.5 * n * (m + 1) + 0.5 * m)) - 1)
                    D_index = np.append(D_index, 2 * indx)
                    D_index = np.append(D_index, 2 * indx + 1)
                elif args.mode == 3:
                    # node_loc=(0.5*n, 0.5*m)
                    indx = np.append(indx, int((0.5 * n * (m + 1))) - 1)
                    D_index = np.append(D_index, 2 * indx)
                    D_index = np.append(D_index, 2 * indx + 1)
                elif args.mode == 4:
                    # for i in range(n):
                    # indx = np.append(indx, int((0.5 * n * (m + 1))) - 1)
                    indx = np.append(indx, int((0.25 * n * (m + 1) + 0.25 * m)) - 1)
                    D_index = np.append(D_index, 2 * indx)
                    D_index = np.append(D_index, 2 * indx + 1)
                elif args.mode == 5:
                    # for i in range(n):
                    indx = np.append(indx, int((0.5 * n * (m + 1))) - 1)
                    indx = np.append(indx, int((0.25 * n * (m + 1) + 0.25 * m)) - 1)
                    D_index = np.append(D_index, 2 * indx)
                    D_index = np.append(D_index, 2 * indx + 1)
                elif args.mode == 6:
                    # for i in range(n):
                    indx = np.append(indx, int((0.5 * n * (m + 1))) - 1)
                    indx = np.append(indx, int((0.25 * n * (m + 1) + 0.25 * m)) - 1)
                    D_index = np.append(D_index, 2 * indx)
                    D_index = np.append(D_index, 2 * indx + 1)
            if args.domain == "building":
                if args.mode == 1:
                    indx = np.append(indx, int((n * m - m * 0.5)))
                    D_index = np.append(D_index, 2 * indx)
                elif args.mode == 2:
                    indx = np.append(indx, int(58200))
                    D_index = np.append(D_index, 2 * 58200 + 1)
                elif args.mode == 3:
                    for i in range(n):
                        indx = np.append(indx, i * m)
                        indx = np.append(indx, i * m + m - 1)
                        # D_index = np.append(D_index, 2 * (i * m))
                        D_index = np.append(D_index, 2 * (i * m + m - 1))
                elif args.mode == 4:
                    for i in range(n):
                        indx = np.append(indx, i * m + m // 2)
                        D_index = np.append(D_index, 2 * (i * m + m // 2))

    domain.visualize(args.prefix, X, bcs, non_design_nodes, forces, indx)

    if args.kokkos:
        import kokkos as kokkos

        kokkos.initialize()

    # Create the filter
    fltr = NodeFilter(
        conn, X, r0, ftype=args.filter, beta=args.beta0, projection=args.proj
    )
    # Create analysis
    analysis = TopologyAnalysis(
        fltr,
        conn,
        X,
        bcs,
        forces,
        D_index,
        fun=args.fun,
        N_a=args.N_a,
        N_b=args.N_b,
        N=args.N,
        atype=args.atype,
        ptype_K=args.ptype_K,
        ptype_M=args.ptype_M,
        rho0_K=args.rho0_K,
        rho0_M=args.rho0_M,
        rho0_G=args.rho0_G,
        p=args.p,
        q=args.q,
        epsilon=args.stress_relax,
        assume_same_element=args.assume_same_element,
        check_gradient=args.check_gradient,
        check_kokkos=args.check_kokkos,
        prob=args.problem,
        kokkos=args.kokkos,
    )

    if args.compliance_ub_percent is not None:
        args.compliance_ub_percent = args.compliance_ub_percent * args.min_compliance

    # Create optimization problem
    topo = TopOptProb(
        analysis,
        non_design_nodes,
        ks_rho_natural_freq=args.ks_rho_natural_freq,
        ks_rho_buckling=args.ks_rho_buckling,
        ks_rho_stress=args.ks_rho_stress,
        ks_rho_freq=args.ks_rho_freq,
        ks_rho_dis_max=args.ks_rho_dis_max,
        m0=args.m0,
        draw_every=5,
        prefix=args.prefix,
        dv_mapping=dv_mapping,
        lb=args.lb,
        prob=args.problem,
        objf=args.objf,
        confs=args.confs,
        omega_lb=args.omega_lb,
        BLF_lb=args.BLF_lb,
        stress_ub=args.stress_ub,
        compliance_ub=args.compliance_ub_percent,
        vol_frac_ub=args.vol_frac_ub,
        vol_frac_lb=args.vol_frac_lb,
        frequency_scale=args.frequency_scale,
        sigma_fixed=args.sigma_fixed,
        sigma_scale=args.sigma_scale,
        stress_scale=args.stress_scale,
        compliance_scale=args.compliance_scale,
        weight=args.weight,
        c0=args.c0,
        mu_ks0=args.mu_ks0,
        dis_ub=args.dis_ub,
        dis_ub_frac=args.dis_ub_frac,
        check_gradient=args.check_gradient,
        domain=args.domain,
        iter_crit=args.iter_crit,
        iter_crit_dis=args.iter_crit_dis,
        iter_crit_stress=args.iter_crit_stress,
        iter_crit_w=args.iter_crit_w,
        iter_crit_d=args.iter_crit_d,
        restart_beta_p=args.restart_beta_p,
        delta_beta=args.delta_beta,
        delta_p=args.delta_p,
        tracking=args.tracking,
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
        if args.check_gradient:
            np.random.seed(0)
            for i in range(5):
                mmaopt.checkGradients()
            exit(0)

        mmaopt.optimize(niter=args.maxit, verbose=False, movelim=args.movelim)
        xopt = mmaopt.getOptimizedDesign()

    else:
        from mpi4py import MPI

        paroptprob = ParOptProb(MPI.COMM_SELF, topo)
        dh = [1e-6]
        if args.check_gradient:
            for i in range(len(dh)):
                # Note: the accuracy highly depends on the value N_a and N_b
                gradient_check(analysis, args.problem)
                paroptprob.checkGradients(dh[i])
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

    if args.kokkos:
        kokkos.finalize()


if __name__ == "__main__":
    args = settings.parse_cmd_args()
    main(args)
