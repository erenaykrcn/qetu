# Helper functions.
from scipy.special import erf
import numpy as np


def crandn(size=None, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


def filter_function(w: float, a: float, da: float, b: float, db: float):
    """
    Filter function in Fourier representation, Eq. (E2).
    """
    return 0.5 * (erf((w + a) / da) - erf((w + b) / db))

def construct_shoveling_lindblad_operator(A, H, f):
    """
    Construct the "shoveling" Lindblad operator, Eq. (4).
    """
    A = np.asarray(A)
    H = np.asarray(H)

    # diagonalize Hamiltonian
    w, v = np.linalg.eigh(H)

    # construct K operator, as in frequency domain.
    return sum((f(w[i] - w[j]) * np.vdot(v[:, i], A @ v[:, j]))
               * np.outer(v[:, i], v[:, j].conj())  # ket-bra product.
                   for i in range(len(w))
                   for j in range(len(w)))