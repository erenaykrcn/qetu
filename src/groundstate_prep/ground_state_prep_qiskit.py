import qiskit
import h5py
import os
import numpy as np
from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate
from qiskit.circuit.library import StatePreparation
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.providers.aer.noise import NoiseModel, errors

from ground_state_prep import get_phis
from utils_gsp import construct_ising_local_term, approx_polynomial, get_phis, qc_U, qc_U_Strang

decomp = TwoQubitBasisDecomposer(gate=CXGate())

# We demonstrate that we can implement the QETU Circuit with the TFIM Hamiltonian
# without including a controlled time evolution operator. This can be achieved by inserting controlled-
# single qubit gates, before and after the time evolution gate. To implement the conjugated time evolution
# operator, we insert X gates to the ancilla qubit, before and after the controlled-time evolution gate.


def get_error_from_sv(qc, qc_H, depolarizing_error, reps, L, J, g, ev, nshots=1e5, t1=3e10, t2=3e10, gate_t=100):
    backend = Aer.get_backend("aer_simulator")
    backend._configuration.max_shots = 1e10
    
    x1_error = errors.depolarizing_error(depolarizing_error*0.1, 1)
    x2_error = errors.depolarizing_error(depolarizing_error, 2)
    relax_error = errors.thermal_relaxation_error(t1, t2, gate_t)
    x1_error = x1_error.compose(relax_error)
    x2_error = x2_error.compose(relax_error)
    no_error = errors.depolarizing_error(0, L+1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(x1_error, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(x2_error, ['cx', 'cy', 'cz', 'unitary'])
    noise_model.add_all_qubit_quantum_error(no_error, ['meas0'])
    noise_model.add_basis_gates(['unitary', 'meas0'])
    
    E_nps = []
    for i in range(reps):
        print("getting counts")
        counts_dict1 = execute(transpile(qc), backend, noise_model=noise_model, basis_gates=noise_model.basis_gates, shots=nshots).result().get_counts()
        counts_dict2 = execute(transpile(qc_H), backend, noise_model=noise_model, basis_gates=noise_model.basis_gates, shots=nshots).result().get_counts()
        print("gotten counts")
        
        E_np = estimate_eigenvalue_counts(counts_dict1, counts_dict2, L, J, g)
        E_nps.append(E_np)
        print(E_np)
    E_nps = np.array(E_nps)

    return np.linalg.norm(np.sum(E_nps)/E_nps.size - ev)


def prepare_ground_state_qiskit(L, J, g, t, mu, a_values, c2=0, repeat_qetu=3, RQC_layers=3, init_state=None):
    V_list = []
    dirname = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dirname, f"../rqcopt/results/ising1d_L{L}_t{t}_dynamics_opt_layers{RQC_layers}.hdf5")

    with h5py.File(path, "r") as f:
        assert f.attrs["L"] == L
        assert f.attrs["J"] == J
        assert f.attrs["g"] == g
        assert f.attrs["t"] == t
        V_list = list(f["Vlist"])
    perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(V_list))]

    print("Running decomposition of two-qubit gates of the RQC Circuit...")
    qcs_rqc = []
    for V in V_list:
        #decomp = TwoQubitBasisDecomposer(gate=CXGate())
        #qc_rqc = decomp(V)
        qc_rqc = qiskit.QuantumCircuit(2)
        qc_rqc.unitary(V, [0, 1])
        qcs_rqc.append(qc_rqc)


    backend = Aer.get_backend("statevector_simulator")
    phis = get_phis(mu, 18, 0.005, 0.9)
    print("F(a_max)^2: ", phis[2](a_values[0])**2)

    qc_U_ins = qc_U(qcs_rqc, L, perms)
    qc_QETU = qc_QETU_cf_R(qc_U_ins, phis[0], c2)
    qc_ins = qiskit.QuantumCircuit(L+1, L+1)

    statevectors_bR = []
    statevectors_aR = []

    if init_state is not None:
        qc_ins.initialize(init_state)
    for i in range(repeat_qetu):
        print("Layer ", i)
        qc_ins.append(qc_QETU.to_gate(), [i for i in range(L+1)])
        bR = execute(transpile(qc_ins), backend).result().get_statevector().data
        statevectors_bR.append(bR)
        # TODO: This workaround should be only temporary! Reset is not working as expected!
        #qc_ins.reset(L)
        #aR = execute(transpile(qc_ins), backend).result().get_statevector()
        aR = np.kron(np.array([[1,0],[0,0]]), np.identity(2**L)) @ bR
        aR = aR / np.linalg.norm(aR)
        statevectors_aR.append(aR)
        statePrep_Gate = StatePreparation(aR, label='meas0')
        qc_ins.reset([i for i in range(L+1)])
        qc_ins.append(statePrep_Gate, [i for i in range(L+1)])

    qc_ins2 = qc_ins.copy()
    qc_ins2.h([i for i in range(L)])
    qc_ins.measure([i for i in range(L+1)], [i for i in range(L+1)])
    qc_ins2.measure([i for i in range(L+1)], [i for i in range(L+1)])

    qc_RQC = qc_ins.copy()
    qc_H_RQC = qc_ins2.copy()

    backend = Aer.get_backend("statevector_simulator")
    qc_U_ins = qc_U_Strang(L, J, g, t, 1)
    qc_QETU = qc_QETU_cf_R(qc_U_ins, phis[0], c2)
    qc_ins = qiskit.QuantumCircuit(L+1, L+1)

    if init_state is not None:
        qc_ins.initialize(init_state)
    for i in range(repeat_qetu):
        print("Layer ", i)
        qc_ins.append(qc_QETU.to_gate(), [i for i in range(L+1)])
        bR = execute(transpile(qc_ins), backend).result().get_statevector().data
        statevectors_bR.append(bR)
        aR = np.kron(np.array([[1,0],[0,0]]), np.identity(2**L)) @ bR
        aR = aR / np.linalg.norm(aR)
        statevectors_aR.append(aR)
        statePrep_Gate = StatePreparation(aR, label='meas0')
        qc_ins.reset([i for i in range(L+1)])
        qc_ins.append(statePrep_Gate, [i for i in range(L+1)])
        #qc_ins.reset(L)

    qc_ins2 = qc_ins.copy()
    qc_ins2.h([i for i in range(L)])
    qc_ins.measure([i for i in range(L+1)], [i for i in range(L+1)])
    qc_ins2.measure([i for i in range(L+1)], [i for i in range(L+1)])

    qc_STR = qc_ins.copy()
    qc_H_STR = qc_ins2.copy()
    return qc_RQC, qc_H_RQC, qc_STR, qc_H_STR



def Pr_counts(counts, j, zj, *zjplus1):
    # Helper Function to retrieve the probability of measuring the qubit number j (and optionally j+1)
    prob = 0
    nshots = 0
    for key, value in counts.items():
        nshots += value
    
    for bitstring, count in counts.items():
        bitstring = bitstring[::-1]
        bitstring = bitstring[:-1]
        
        L = len(bitstring)
        
        if not zjplus1:
            if bitstring[j] == str(zj):
                prob += count/nshots
        else:
            if j == L-1:
                if bitstring[j] == str(zj) and bitstring[0] == str(zjplus1[0]):
                    prob += count/nshots
            else:
                if bitstring[j] == str(zj) and bitstring[j+1] == str(zjplus1[0]):
                    prob += count/nshots
    return prob


def estimate_eigenvalue_counts(counts1, counts2, L, J, g):
    E1 = 0
    # For consistency with qib, upper range has to be L!
    for j in range(L):
        for zj in range(2):
            for zjplus1 in range(2):
                E1 = E1 + (((-1)**(zj + zjplus1)) * Pr_counts(counts1, j, zj, zjplus1))
    
    E2 = 0
    for j in range(L):
        for zj in range(2):
            E2 = E2 + (((-1)**zj) * Pr_counts(counts2, j, zj))

    return J*E1 + g*E2


def qc_cfU_R(qc_U, dagger, c2):
    L = qc_U.num_qubits
    qc_cfU = qiskit.QuantumCircuit(L+1)
    
    if dagger:
        qc_cfU.x(L)
    
    qc_cfU.x(L)
    for j in range(L-1, -1, -1):
        if j % 2 == 1:
            qc_cfU.cy(L, j)
        else:
            qc_cfU.cz(L, j)
    qc_cfU.x(L)
    
    qc_cfU.append(qc_U.to_gate(), [i for i in range(L)])
    
    qc_cfU.x(L)
    for j in range(L-1, -1, -1):
        if j % 2 == 1:
            qc_cfU.cy(L, j)
        else:
            qc_cfU.cz(L, j)
    qc_cfU.x(L)

    qc_cfU.cp(-c2, L, 0)
    qc_cfU.x(0)
    qc_cfU.cp(-c2, L, 0)
    qc_cfU.x(0)
    
    if dagger:
        qc_cfU.x(L)
    
    return qc_cfU

def qc_QETU_cf_R(qc_U, phis, c2=0):
    L = qc_U.num_qubits
    qc = qiskit.QuantumCircuit(L+1)
    qc.rx(-2*phis[0], L)
    for i in range(1, len(phis)):
        qc.append(qc_cfU_R(qc_U, i%2==0, c2).to_gate(), [i for i in range(L+1)])
        qc.rx(-2*phis[i], L)
    return qc