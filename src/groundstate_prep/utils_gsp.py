import qiskit
import numpy as np
from scipy.linalg import expm
import cvxpy as cp
from numpy import polynomial as P
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
import rqcopt as oc
import scipy

# Below are the functions to randomize a diagonal hamiltonian
# and QETU circuit using directly controlled time evolution block.
# Implemented with numpy.

unnorm_Hadamard = np.array([[1, 1],[1, -1]])
ket_0 = np.array([1, 0])

def H(a, dim):
    if dim == 0:
        return 2*np.arccos(a)
    H = np.identity(dim)
    H[0][0] = 2*np.arccos(a)
    
    ai = 0.1
    for i in range(1, dim):
        ai = ai - 1e-8
        H[i][i] = 2*np.arccos(ai)
    return H

def get_a(a_max_arg=None, a_premax_arg=None, cos_eta_2=0.6):
    if a_max_arg:
        a_max = a_max_arg
    else:
        a_max = random() * 0.3 + cos_eta_2
    
    if a_premax_arg:
        a_premax = a_premax_arg
    else:
        a_premax = random()*0.05

    print("a_premax: " + str(a_premax))
    print("a_max: " + str(a_max))
    return a_max, a_premax

def S(phi, dim):
    S_z = np.array([[np.exp(1j*phi), 0],[0, np.exp(-1j*phi)]])
    S_x =  0.5*(unnorm_Hadamard @ S_z @ unnorm_Hadamard)
    return np.kron(S_x, np.identity(dim))

def cU(H, dagger, tau=1, shift=0):
    offset = len(H)
    dim = 2*offset
    cU  = (1+0j) * np.identity(dim)
    
    H = tau*H + shift*np.identity(offset)
    
    expH = expm(-1j * H)
    if dagger:
        expH = expH.transpose().conjugate()
        
    for i in range(offset):
        for j in range(offset):
            cU[offset+i][offset+j] = expH[i][j]
    return cU

def QETU(H, phis, tau=1, shift=0):
    U = S(phis[len(phis)-1], len(H))
    for i in range(1, len(phis)):
        U = U @ cU(H, i % 2 == 1, tau, shift) @ S(phis[len(phis)-1-i], len(H))
    return U


# Helper functions for visualizing.
def W_x(a):
    return np.array([[a, 1j*np.sqrt(1-a**2)], [1j*np.sqrt(1-a**2), a]])

def S_1(phi):
    return np.array([[np.exp(1j*phi), 0],[0, np.exp(-1j*phi)]])

def U(phi, a):
    U = S_1(phi[0])
    for i in range(1, len(phi)):
        U = U @ W_x(a) @ S_1(phi[i])
    return U


def construct_ising_local_term(J, g):
    """
    Construct local interaction term of Ising Hamiltonian on a one-dimensional
    lattice for interaction parameter `J` and external field parameter `g`.
    """
    # Pauli-X and Z matrices
    X = np.array([[0.,  1.], [1.,  0.]])
    Z = np.array([[1.,  0.], [0., -1.]])
    I = np.identity(2)
    return J*np.kron(Z, Z) + g*0.5*(np.kron(X, I) + np.kron(I, X))


# Quantum Signal Processing step to optimize the phi values.
def get_phis(x, d, h, c=0.99):
    assert d % 2 == 0
    a_list = np.linspace(-1, 1, 101)
    
    poly_even_step, even_coeffs = approx_polynomial(500, d, x-h, x+h, c, 0.01)
    poly_even_step  = poly_even_step.convert(kind=P.Polynomial)
    
    ket_0 = np.array([1,0])
    
    # TODO: Degrees of freedom is actually d/2! With that we can get much smoother functions!
    phi_primes    = QuantumSignalProcessingPhases(poly_even_step, signal_operator="Wx", method="laurent")
    phi_primes[0] = (phi_primes[-1] + phi_primes[0])/2
    phi_primes[-1] = phi_primes[0]
    
    """qsp_polynom = [poly_even_step(a) for a in a_list]
    plt.plot(a_list, qsp_polynom, label="Polynom")
    qsp_polynom = [np.vdot(ket_0, U(phi_primes, a)@ket_0).real for a in a_list]
    plt.plot(a_list, qsp_polynom, "--", label=r"$Re{\langle0|U_{\phi}|0\rangle}$")
    qsp_polynom = [np.vdot(ket_0, U(phi_primes, a)@ket_0).imag for a in a_list]
    plt.plot(a_list, qsp_polynom, "--", label=r"$Im{\langle0|U_{\phi}|0\rangle}$")
    plt.legend()"""
    
    phis = [phi_prime + np.pi/4 if i==0 or i==len(phi_primes)-1 else phi_prime + np.pi/2 for i, phi_prime in enumerate(phi_primes)]
    return (phis, phi_primes, poly_even_step)


# Functions for approximating the even step function with polynomials.
def Cheby_polyf(x: [float], k: int):
    x = np.array(x)
    y = np.arccos(x)
    ret = np.cos((2*k)*y)
    return ret


def approx_polynomial(M, d, sigma_minus, sigma_plus, c, eps, x_list_arg=None):
    assert d%2 == 0
    x = []
    for j in range(M):
        xj = -np.cos(j*np.pi/(M-1))
        if np.abs(xj)<= sigma_minus or np.abs(xj) >= sigma_plus:
            x.append(xj)
    
    def cost_func(x_c, coeff, c):
        A = []
        for k in range(int(coeff.shape[0])):
            A.append(Cheby_polyf(x_c, k))
        A = np.array(A)
        A = A.transpose()
        
        b = [0 for _ in range(len(x_c))]
        for j, xj in enumerate(x_c):
            if np.abs(xj) <= sigma_minus:
                b[j] = 0
            else:
                b[j] = c
        b = np.array(b)
        return (A @ coeff) - b
    
    x_list = np.linspace(-1, 1, 101)
    if x_list_arg:
        x_list = x_list_arg
    
    coeff = cp.Variable(int(d/2))
    constraints = [np.sum([ck*Cheby_polyf([x_i], k)[0] for k,ck in enumerate(coeff)]) <= c-eps for x_i in x_list]
    constraints += [np.sum([ck*Cheby_polyf([x_i], k)[0] for k,ck in enumerate(coeff)]) >= eps for x_i in x_list]

    objective = cp.Minimize(cp.sum_squares(cost_func(x, coeff, c)))
    
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    coeffs = coeff.value

    func = [(  np.sum([ck*Cheby_polyf([x], k)[0] for k,ck in enumerate(coeffs)])  ) for x in x_list]
    for f in func:
        if f < 0 or f > 1: 
            raise Exception("Found polynomial exceeds the [0,1] Interval!")
    
    return P.chebyshev.Chebyshev([coeffs[int(i/2)] if i%2==0 else 0 for i in range(2*len(coeffs)-1)]), coeffs


def qc_U(two_qubit_gates, L, perms):
    U = qiskit.QuantumCircuit(L)
    for layer, qc_gate in enumerate(two_qubit_gates):
        assert L%2 == 0
        for j in range(L//2):
            if perms[layer] is not None:
                U.append(qc_gate.to_gate(), [perms[layer][2*j], perms[layer][2*j+1]])
            else:
                U.append(qc_gate.to_gate(), [2*j, 2*j+1])
    return U


def qc_U_Strang(L, J, g, t, nsteps):
    U = qiskit.QuantumCircuit(L)
    
    dt = t/nsteps
    hloc = construct_ising_local_term(J, g)
    coeffs = oc.SplittingMethod.suzuki(2, 1).coeffs
    #strang = oc.SplittingMethod.suzuki(2, 1)
    #_, coeffs = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
    #coeffs = [0.5*c for c in coeffs]

    Vlist = [scipy.linalg.expm(-1j*c*dt*hloc) for c in coeffs]
    Vlist_gates = []
    for V in Vlist:
        #decomp = TwoQubitBasisDecomposer(gate=CXGate())
        #qc = decomp(V)
        qc = qiskit.QuantumCircuit(2)
        qc.unitary(V, [0, 1], label='str')
        Vlist_gates.append(qc)
    perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(coeffs))]

    for layer, qc_gate in enumerate(Vlist_gates):
        assert L%2 == 0
        for j in range(L//2):
            if perms[layer] is not None:
                U.append(qc_gate.to_gate(), [perms[layer][2*j], perms[layer][2*j+1]])
            else:
                U.append(qc_gate.to_gate(), [2*j, 2*j+1])
    return U
    

    
