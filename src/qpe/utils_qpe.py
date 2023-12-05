import qiskit
import scipy
import numpy as np
from qiskit.circuit.library import StatePreparation
from qiskit import Aer, transpile, execute


def qc_QPE(L, initial_state, qc_cU):
    qpe_real = qiskit.QuantumCircuit(L+1, 1)
    qpe_real.initialize(initial_state)
    qpe_real.h(L)
    qpe_real.append(qc_cU.to_gate(), [i for i in range(L+1)])
    qpe_imag = qpe_real.copy()
    qpe_imag.p(-0.5*np.pi, L)
    qpe_imag.h(L)
    qpe_real.h(L)
    qpe_real.measure(L, 0)
    qpe_imag.measure(L, 0)
    return qpe_real, qpe_imag


def qc_QPE_noisy_sim(L, qc_qetu, qc_cU, qetu_repeat=3):
    qpe_real = qiskit.QuantumCircuit(L+1, 1)

    backend = Aer.get_backend("statevector_simulator")
    for i in range(qetu_repeat):
        qpe_real.append(qc_qetu.to_gate(), [i for i in range(L+1)])
        bR = execute(transpile(qpe_real), backend).result().get_statevector().data
        aR = np.kron(np.array([[1,0],[0,0]]), np.identity(2**L)) @ bR
        aR = aR / np.linalg.norm(aR)
        statePrep_Gate = StatePreparation(aR, label='meas0')
        qpe_real.reset([i for i in range(L+1)])
        qpe_real.append(statePrep_Gate, [i for i in range(L+1)])

    qpe_real.h(L)
    qpe_real.append(qc_cU.to_gate(), [i for i in range(L+1)])
    qpe_imag = qpe_real.copy()
    qpe_imag.p(-0.5*np.pi, L)
    qpe_imag.h(L)
    qpe_real.h(L)
    qpe_real.measure(L, 0)
    qpe_imag.measure(L, 0)
    return qpe_real, qpe_imag


def controlled_trotterized_time_evolution(qc, coeffs, hloc, dt, L):
    Vlist = [scipy.linalg.expm(-1j*c*dt*hloc) for c in coeffs]
    Vlist_gates = []
    for V in Vlist:
        qc2 = qiskit.QuantumCircuit(2)
        qc2.unitary(V, [0, 1], label='str')
        Vlist_gates.append(qc2)
    perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(coeffs))]
    for layer, qc_gate in enumerate(Vlist_gates):
        for j in range(L//2):
            if perms[layer] is not None:
                qc.append(qc_gate.to_gate().control(), [L, L-(perms[layer][2*j]+1), L-(perms[layer][2*j+1]+1)])
            else:
                qc.append(qc_gate.to_gate().control(), [L, L-(2*j+1), L-(2*j+2)])


def construct_ising_local_term(J, g):
    X = np.array([[0.,  1.], [1.,  0.]])
    Z = np.array([[1.,  0.], [0., -1.]])
    I = np.identity(2)
    return J*np.kron(Z, Z) + g*0.5*(np.kron(X, I) + np.kron(I, X))