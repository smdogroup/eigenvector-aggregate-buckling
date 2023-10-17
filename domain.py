import os
import shutil as shutil

from icecream import ic
import matplotlib.pylab as plt
import numpy as np
from scipy.sparse import coo_matrix, linalg


def cantilever(lx=20, ly=10, m=128, n=64):
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


def lbracket(r0_=2.1, l=8.0, lfrac=0.4, nx=96, m0_block_frac=0.0):
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


def beam(r0_=2.1, l=8.0, frac=0.125, nx=100, prob="natural_frequency"):
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
    forces = {}
    if prob == "natural_frequency":
        # fix the middle left and right
        bcs[nodes[n // 2, 0]] = [0, 1]
        bcs[nodes[n // 2, m]] = [0, 1]
    elif prob == "buckling":
        # fix the bottom left and right
        offset = int(np.ceil(m * 0.02))
        for i in range(offset):
            bcs[nodes[0, i]] = [0, 1]
            bcs[nodes[0, m - i]] = [0, 1]

        # force is independent of the mesh size apply a force at the top middle
        P = 100.0
        offset = int(np.ceil(m / 40))
        for i in range(offset):
            forces[nodes[n, m // 2 - i]] = [0, -P / (2 * offset)]
            forces[nodes[n, m // 2 + 1 + i]] = [0, -P / (2 * offset)]

    r0 = l / nx * r0_
    ic(r0)

    Ei = []
    Ej = []
    redu_idx = 0

    if prob == "natural_frequency":
        # 4-way reflection of x- and y-symmetry axes
        a = n // 2
        b = m // 2
        for i in range(a + 1):
            for j in range(b + 1):
                if nodes[i, j] not in non_design_nodes:
                    Ej.extend(4 * [redu_idx])
                    Ei.extend(
                        [
                            nodes[i, j],
                            nodes[n - i, j],
                            nodes[i, m - j],
                            nodes[n - i, m - j],
                        ]
                    )
                    redu_idx += 1
    elif prob == "buckling":
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


def rhombus(r0_=2.1, l=2.0, frac=3, nx=100):
    """
    __________________________|
    |                         |
    |                         | n = nx
    |_________________________|
            m = 3 * nx
    """

    m = int(frac * nx)
    n = nx

    nelems = m * n
    nnodes = (m + 1) * (n + 1)

    y = np.linspace(0, l, n + 1)
    x = np.linspace(0, l * frac, m + 1)
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
    forces = {}
    # fix the bottom left and right
    offset = int(np.ceil(n / 20))
    for i in range(offset):
        bcs[nodes[0, i]] = [0, 1]
        bcs[nodes[0, int(2 * n) + i]] = [1]

    # bcs[nodes[0, 0]] = [0, 1]
    # bcs[nodes[0, int(2*n)]] = [1]

    # force is independent of the mesh size apply a force at the top middle
    P = 1000.0
    offset = int(np.ceil(n / 20))
    # offset = 1
    for i in range(offset):
        forces[nodes[n, m - i]] = [0, -P / offset]

    r0 = l / nx * r0_
    ic(r0)

    return conn, X, r0, bcs, forces, non_design_nodes


def building(r0_=2.1, l=1.0, frac=2, nx=100, m0_block_frac=0.0):
    """
    _______
    |     |
    |     |
    |     | n
    |     |
    |_____|
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

    ic(nodes.T.shape)

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
    # m0_block_frac = 0.25
    # nm = int(np.ceil(m * m0_block_frac))
    # if nm % 2 == 1:
    #     nm -= 1
    # nm = 2
    # offset = int(np.floor(m / 20))
    # nm = 2 * offset

    # for i in range(n - int(nm/2) + 1, n + 1):
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
    for j in range(m + 1):
        bcs[nodes[0, j]] = [0, 1]

    # bcs[nodes[0, 0]] = [0, 1]
    # bcs[nodes[0, m]] = [0, 1]

    # force is independent of the mesh size
    P = 1e-3
    forces = {}
    # apply a force at the top middle
    offset = int(np.ceil(m / 30))
    for i in range(offset):
        forces[nodes[n, m // 2 - i]] = [0, -P / (2 * offset)]
        forces[nodes[n, m // 2 + 1 + i]] = [0, -P / (2 * offset)]

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


def leg(r0_=2.1, l=8.0, frac=2, nx=100, m0_block_frac=0.0):
    """
    _______
    |     |
    |     |
    |     | n
    |     |
    |_____|
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

    ic(nodes.T.shape)
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

    offset = int(np.ceil(m / 10))
    if offset % 2 == 1:
        offset -= 1  # make sure offset is even

    # three square blocks as non-design region
    for i in range((n - offset) // 2 + 1, (n + offset) // 2 + 1):
        for j in range(m - offset + 1, m + 1):
            non_design_nodes.append(nodes[i, j])

    for i in range(n - offset + 1, n + 1):
        for j in range(0, offset):
            non_design_nodes.append(nodes[i, j])
    for i in range(0, offset):
        for j in range(0, offset):
            non_design_nodes.append(nodes[i, j])

    bcs = {}
    bcs[nodes[0, 0]] = [0, 1]
    bcs[nodes[n, 0]] = [0, 1]

    # force is independent of the mesh size
    P = 1000.0
    forces = {}
    # apply a force at the right middle edge
    offset = offset // 2
    for i in range(offset):
        forces[nodes[n // 2 - i, m]] = [0, -P / (2 * offset)]
        forces[nodes[n // 2 + 1 + i, m]] = [0, -P / (2 * offset)]

    r0 = l / nx * r0_
    ic(r0)

    return conn, X, r0, bcs, forces, non_design_nodes


def square(r0_, l=1.0, nx=30, m0_block_frac=0.0):
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

    # for i in range(offset):
    #     bcs[nodes[0, i]] = [1]
    #     bcs[nodes[0, m - i]] = [1]
    #     bcs[nodes[n, i]] = [1]
    #     bcs[nodes[n, m - i]] = [1]

    # for j in range(offset):
    #     bcs[nodes[j, 0]] = [0]
    #     bcs[nodes[j, m]] = [0]
    #     bcs[nodes[n - j, 0]] = [0]
    #     bcs[nodes[n - j, m]] = [0]

    # fix the bottom left corner
    bcs[nodes[0, 0]] = [0, 1]
    # fix the bottom right corner
    # bcs[nodes[0, m]] = [0, 1]
    bcs[nodes[0, m]] = [1]
    # # fix the top left corner
    # bcs[nodes[n, 0]] = [0, 1]
    bcs[nodes[n, 0]] = [0]
    # # fix the top right corner
    # bcs[nodes[n, m]] = [0, 1]

    P = 2e-3
    forces = {}
    # apply force for the four sides uniformly
    for i in range(nodes.shape[1]):
        # forces[nodes[0, i]] = [0, P / nodes.shape[1]]
        forces[nodes[n, i]] = [0, -P / nodes.shape[1]]
        # forces[nodes[i, 0]] = [P / nodes.shape[1], 0]
        forces[nodes[i, m]] = [-P / nodes.shape[1], 0]

    # forces[nodes[n, 0]] = [P, -P]
    # forces[nodes[n, m]] = [-P, -P]
        
    
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


# def square(r0_, l=1.0, nx=30, m0_block_frac=0.0):
#     """
#     Args:
#         l: length of the square
#         nx: number of elements along x direction
#     """

#     # Generate the square domain problem by default
#     m = nx
#     n = nx

#     nelems = m * n
#     nnodes = (m + 1) * (n + 1)

#     y = np.linspace(0, l, n + 1)
#     x = np.linspace(0, l, m + 1)
#     nodes = np.arange(0, (n + 1) * (m + 1)).reshape((n + 1, m + 1))

#     # Set the node locations
#     X = np.zeros((nnodes, 2))
#     for j in range(n + 1):
#         for i in range(m + 1):
#             X[i + j * (m + 1), 0] = x[i]
#             X[i + j * (m + 1), 1] = y[j]

#     # Set the connectivity
#     conn = np.zeros((nelems, 4), dtype=int)
#     for j in range(n):
#         for i in range(m):
#             conn[i + j * m, 0] = nodes[j, i]
#             conn[i + j * m, 1] = nodes[j, i + 1]
#             conn[i + j * m, 2] = nodes[j + 1, i + 1]
#             conn[i + j * m, 3] = nodes[j + 1, i]

#     # We would like the center node or element to be the non-design region
#     non_design_nodes = []
#     # offset = int(m0_block_frac * nx * 0.5)
#     # for j in range(n // 2 - offset, (n + 1) // 2 + 1 + offset):
#     #     for i in range(n // 2 - offset, (n + 1) // 2 + 1 + offset):
#     #         non_design_nodes.append(nodes[j, i])

#     # Constrain all boundaries
#     bcs = {}

#     offset = int(nx * 0.1)

#     for i in range(offset):
#         bcs[nodes[0, i]] = [1]
#         bcs[nodes[0, m - i]] = [1]
#         bcs[nodes[n, i]] = [1]
#         bcs[nodes[n, m - i]] = [1]

#     for j in range(offset):
#         bcs[nodes[j, 0]] = [0]
#         bcs[nodes[j, m]] = [0]
#         bcs[nodes[n - j, 0]] = [0]
#         bcs[nodes[n - j, m]] = [0]

#     # fix the bottom left corner
#     bcs[nodes[0, 0]] = [0, 1]
#     # fix the bottom right corner
#     bcs[nodes[0, m]] = [0, 1]
#     # fix the top left corner
#     bcs[nodes[n, 0]] = [0, 1]
#     # fix the top right corner
#     bcs[nodes[n, m]] = [0, 1]

#     P = 1e-3
#     forces = {}
#     # # apply force for the four sides uniformly
#     # for i in range(nodes.shape[1]):
#     #     forces[nodes[0, i]] = [0, P / nodes.shape[1]]
#     #     forces[nodes[n, i]] = [0, -P / nodes.shape[1]]
#     #     forces[nodes[i, 0]] = [P / nodes.shape[1], 0]
#     #     forces[nodes[i, m]] = [-P / nodes.shape[1], 0]
    
#     # apply force at the center in x, y direction
#     forces[nodes[n//2, m//2]] = [P, P]
#     forces[nodes[n//2, m//2]] = [-P, -P]
#     # pn = n // 10
#     # for j in range(pn):
#     #     forces[nodes[j, -1]] = [0, -P / pn]

#     r0 = l / nx * r0_

#     # Create the mapping E such that x = E*xr, where xr is the nodal variable
#     # of a quarter and is controlled by the optimizer, x is the nodal variable
#     # of the entire domain
#     Ei = []
#     Ej = []
#     redu_idx = 0

#     # 8-way reflection
#     for j in range(1, (n + 1) // 2):
#         for i in range(j):
#             if nodes[j, i] not in non_design_nodes:
#                 Ej.extend(8 * [redu_idx])
#                 Ei.extend(
#                     [nodes[j, i], nodes[j, m - i], nodes[n - j, i], nodes[n - j, m - i]]
#                 )
#                 Ei.extend(
#                     [nodes[i, j], nodes[i, m - j], nodes[n - i, j], nodes[n - i, m - j]]
#                 )
#                 redu_idx += 1

#     # 4-way reflection of diagonals
#     for i in range((n + 1) // 2):
#         if nodes[i, i] not in non_design_nodes:
#             Ej.extend(4 * [redu_idx])
#             Ei.extend(
#                 [nodes[i, i], nodes[i, m - i], nodes[n - i, i], nodes[n - i, m - i]]
#             )
#             redu_idx += 1

#     # 4-way reflection of x- and y-symmetry axes, only apply if number of elements
#     # along x (and y) is even
#     if n % 2 == 0:
#         j = n // 2
#         for i in range(j + 1):
#             if nodes[i, j] not in non_design_nodes:
#                 Ej.extend(4 * [redu_idx])
#                 Ei.extend([nodes[i, j], nodes[n - i, j], nodes[j, i], nodes[j, n - i]])
#                 redu_idx += 1

#     Ev = np.ones(len(Ei))
#     dv_mapping = coo_matrix((Ev, (Ei, Ej)))

#     return conn, X, r0, bcs, forces, non_design_nodes, dv_mapping


def visualize(prefix, X, bcs, non_design_nodes=None, forces=None, index=None):
    markersize = 1.0
    fig, ax = plt.subplots()
    ax.set_aspect("equal")

    ax.scatter(X[:, 0], X[:, 1], color="black", s=markersize)

    if forces:
        for i, v in forces.items():
            ax.scatter(X[i, 0], X[i, 1], color="orange", s=markersize)
            
    if bcs:
        for i, v in bcs.items():
            if len(v) == 2:
                ax.scatter(X[i, 0], X[i, 1], color="red", s=markersize)
            else:
                ax.scatter(X[i, 0], X[i, 1], color="g", s=markersize)

    if non_design_nodes:
        m0_X = np.array([X[i, :] for i in non_design_nodes])
        ax.scatter(m0_X[:, 0], m0_X[:, 1], color="blue", s=markersize)

    if index:
        m0_X = np.array([X[index, :]])
        ax.scatter(m0_X[:, 0], m0_X[:, 1], color="r", s=2*markersize)

    fig.savefig(os.path.join(prefix, "domain.png"), dpi=500, bbox_inches="tight")
    return


# def building(r0_=2.1, l=1.0, frac=2, nx=100, m0_block_frac=0.0):
#     """
#     _______
#     |     |
#     |     |
#     |     | n
#     |     |
#     |_____|
#        m
#     """

#     m = nx
#     n = int(np.ceil((frac * nx)))

#     # make sure m and n is even
#     if n % 2 == 0:
#         n -= 1
#     if m % 2 == 0:
#         m -= 1

#     nelems = m * n
#     nnodes = (m + 1) * (n + 1)
#     y = np.linspace(0, l * frac, n + 1)
#     x = np.linspace(0, l, m + 1)
#     nodes = np.arange(0, (n + 1) * (m + 1)).reshape((n + 1, m + 1))

#     ic(nodes.T.shape)

#     # Set the node locations
#     X = np.zeros((nnodes, 2))
#     for j in range(n + 1):
#         for i in range(m + 1):
#             X[i + j * (m + 1), 0] = x[i]
#             X[i + j * (m + 1), 1] = y[j]

#     # Set the connectivity
#     conn = np.zeros((nelems, 4), dtype=int)
#     for j in range(n):
#         for i in range(m):
#             conn[i + j * m, 0] = nodes[j, i]
#             conn[i + j * m, 1] = nodes[j, i + 1]
#             conn[i + j * m, 2] = nodes[j + 1, i + 1]
#             conn[i + j * m, 3] = nodes[j + 1, i]

#     non_design_nodes = []
#     # apply top middle a square block
#     offset = int(np.ceil(m / 30))
#     nm = 2 * offset

#     for i in range(n - int(nm / 2) + 1, n + 1):
#         for j in range((m - nm) // 2 + 1, (m + nm) // 2 + 1):
#             non_design_nodes.append(nodes[i, j])

#     # for i in range(n - nm + 1, n + 1):
#     #     for j in range(0, nm):
#     #         non_design_nodes.append(nodes[i, j])
#     # for i in range(n - nm + 1, n + 1):
#     #     for j in range(m - nm + 1, m + 1):
#     #         non_design_nodes.append(nodes[i, j])

#     # for i in range(n - nm + 1, n + 1):
#     #     for j in range(0, m + 1):
#     #         non_design_nodes.append(nodes[i, j])

#     # h = n // 8
#     # for i in range(1, 9):
#     #     for j in range(m + 1):
#     #         non_design_nodes.append(nodes[i * h, j])

#     bcs = {}
#     for j in range(m + 1):
#         bcs[nodes[0, j]] = [0, 1]

#     # bcs[nodes[0, 0]] = [0, 1]
#     # bcs[nodes[0, m]] = [0, 1]

#     # force is independent of the mesh size
#     P = 1e-3
#     forces = {}
#     # apply a force at the top middle
#     for i in range(offset):
#         forces[nodes[n, m // 2 - i]] = [0, -P / nm]
#         forces[nodes[n, m // 2 + 1 + i]] = [0, -P / nm]

#     r0 = l / nx * r0_
#     ic(r0)

#     Ei = []
#     Ej = []

#     # 2-way reflection left to right
#     for j in range(2 * (n + 1)):
#         for i in range((m + 1) // 2):
#             if j % 2 == 0:
#                 Ej.extend([i + j * (m + 1) // 4])
#             else:
#                 Ej.extend([i + (m // 2 - 2 * i) + (j - 1) * (m + 1) // 4])
#             Ei.extend([i + j * (m + 1) // 2])

#     Ev = np.ones(len(Ei))
#     dv_mapping = coo_matrix((Ev, (Ei, Ej)))

#     # change dv_mapping to np.array
#     # dv_mapping = np.array(dv_mapping.todense())
#     # ic(dv_mapping.shape)
#     # ic(dv_mapping)

#     return conn, X, r0, bcs, forces, non_design_nodes, dv_mapping
