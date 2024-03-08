import numpy as np
from scipy.linalg import eigh


def csderiv(func, h=1e-50):
    """Complex-step derivative approximation"""
    return lambda x: np.imag(func(x + 1j * h)) / h


def fdiff(func, h=1e-8):
    """Finite difference derivative approximation"""
    return lambda x: (func(x + h) - func(x - h)) / (2 * h)


def rand_symm_mat(n):
    # Function to generate a random symmetric matrix
    A = np.random.rand(n, n)
    return 0.5 * (A + A.T)


def fun_h(Q):
    # Function to compute h = Q[:, 0]^T * Q[:, 0]
    h = np.dot(Q[:, 0].T, Q[:, 0])
    return h


# Assuming n is defined
n = 3

# Generate random symmetric matrices A, Adot, and Bdot, and identity matrix B
A = rand_symm_mat(n)
Adot = rand_symm_mat(n)

# use scipy.linalg.eigh to compute the eigenvalues and eigenvectors of A
lam, Q = eigh(A)

# compute the derivative of fun_h using complex-step and finite difference methods
fun_h_csderiv = csderiv(lambda x: fun_h(Q + x))
fun_h_fdiff = fdiff(lambda x: fun_h(Q + x))

# compute the derivative of fun_h using complex-step and finite difference methods
h_csderiv = fun_h_csderiv(0)
h_fdiff = fun_h_fdiff(0)

# print the results
print("Complex-step derivative: ", h_csderiv)
print("Finite difference derivative: ", h_fdiff)

# Compare the results
assert np.isclose(h_csderiv, h_fdiff, atol=1e-8)
print("Results are close")

A1 = A + 1e-8 * Adot
lam1, Q1 = eigh(A1)

# compute the derivative of fun_h using complex-step and finite difference methods
fun_h_csderiv = csderiv(lambda x: fun_h(Q1 + x))
fun_h_fdiff = fdiff(lambda x: fun_h(Q1 + x))

# compute the derivative of fun_h using complex-step and finite difference methods
h_csderiv = fun_h_csderiv(0)
h_fdiff = fun_h_fdiff(0)

# print the results
print("Complex-step derivative: ", h_csderiv)
print("Finite difference derivative: ", h_fdiff)

# Compare the results
assert np.isclose(h_csderiv, h_fdiff, atol=1e-8)
print("Results are close")

