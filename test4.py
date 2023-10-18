from icecream import ic
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import mpmath as mp
import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import minimize
from scipy import sparse, spatial


def Eij_a(fun, rho, trace, lam_min, lam1, lam2):
    """
    Compute the precise value of the E_{ij} term

        E_{ij} = exp(-rho * (lam1 - lam_min)) / trace

        if lam1 == lam2:
            E_{ij} = exp(-rho * (lam1 - lam_min)) / trace
        else:
            E_{ij} = (exp(-rho * (lam1 - lam_min)) - exp(-rho * (lam2 - lam_min))) / (lam1 - lam2) / trace
    """

    with mp.workdps(80):
        if lam1 == lam2:
            val = -rho * fun(-rho * (lam1 - lam_min)) / trace
        else:
            val = (
                (fun(-rho * (lam1 - lam_min)) - fun(-rho * (lam2 - lam_min)))
                / (mp.mpf(lam1) - mp.mpf(lam2))
                / mp.mpf(trace)
            )
    return np.float64(val)
  
  
  
a = 10.0
b = 10.0 + 1e-10
d = b - a
Eij = np.exp(a) * (np.expm1(d) / d)
Eij2 = (np.exp(a) - np.exp(b))/(a - b)
Eij4 = Eij_a(np.exp, -1.0, 1.0, 0.0, a, b)
exact = np.exp(b)
print("Eij:  ", Eij, " err: ", (Eij - exact)/exact)
print("Eij2: ", Eij2, " err: ", (Eij2 - exact)/exact)
print("Eij4: ", Eij4, " err: ", (Eij4 - exact)/exact)

ic(mp.tanh(d))
Eij = mp.tanh(a) * (mp.tanh(d) / d)
Eij2 = (mp.tanh(a) - mp.tanh(b))/(a - b)
Eij4 = Eij_a(mp.tanh, -1.0, 1.0, 0.0, a, b)
exact = mp.tanh(b)
ic(mp.tanh(a))
ic(mp.tanh(b))
print("Eij:  ", Eij, " err: ", (Eij - exact)/exact)
print("Eij2: ", Eij2, " err: ", (Eij2 - exact)/exact)
print("Eij4: ", Eij4, " err: ", (Eij4 - exact)/exact)
