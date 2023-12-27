# Demonstrate dataset creation for QCELS Fitting with depolarizing quantum noise.
import qiskit
from qiskit import execute, transpile
import h5py
import rqcopt as oc
from qiskit.providers.aer.noise import NoiseModel, errors
import numpy as np

from utils_qpe import construct_ising_local_term, controlled_trotterized_time_evolution, qc_QPE, qc_QPE_noisy_sim

import sys
sys.path.append("../../src/rqcopt")
from optimize import ising1d_dynamics_opt


def estimate_phases(L, J, g, prepared_state, eigenvalues_sort, tau, N, shots, depolarizing_error, rqc=True, coeffs=None):
    backend = qiskit.Aer.get_backend("aer_simulator")
    x1_error = errors.depolarizing_error(depolarizing_error*0.1, 1)
    x2_error = errors.depolarizing_error(depolarizing_error, 2)
    x3_error = errors.depolarizing_error(depolarizing_error*10, 3)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(x1_error, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(x2_error, ['cu', 'cx','cy', 'cz'])
    noise_model.add_all_qubit_quantum_error(x3_error, ['ccx'])

    phase_estimates_with_noise = []
    phase_exacts = []
    for n in range(1, N):
        t = n*tau
        qc_cU = qiskit.QuantumCircuit(L+1)

        if rqc:
            V_list = []
            path = f"../../src/rqcopt/results/ising1d_L{L}_t{t}_dynamics_opt_layers5.hdf5"
            try:
                with h5py.File(path, "r") as f:
                    assert f.attrs["L"] == L
                    assert f.attrs["J"] == J
                    assert f.attrs["g"] == g
                    V_list = list(f["Vlist"])
            except FileNotFoundError:
                strang = oc.SplittingMethod.suzuki(2, 1)
                _, coeffs_start_n5 = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
                # divide by 2 since we are taking two steps
                coeffs_start_n5 = [0.5*c for c in coeffs_start_n5]
                print("coeffs_start_n5:", coeffs_start_n5)
                V_list = ising1d_dynamics_opt(5, t, False, coeffs_start_n5, path, niter=16)
            perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(V_list))]
            qcs = []
            for V in V_list:
                qc = qiskit.QuantumCircuit(2)
                qc.unitary(V, [0, 1])
                qcs.append(qc)
            for layer, qc_gate in enumerate(qcs):
                cgate = qc_gate.to_gate().control()
                for j in range(L//2):
                    if perms[layer] is not None:
                        qc_cU.append(cgate, [L, L-(perms[layer][2*j]+1), L-(perms[layer][2*j+1]+1)])
                    else:
                        qc_cU.append(cgate, [L, L-(2*j+1), L-(2*j+2)])
        else:
            hloc = construct_ising_local_term(J, g)
            controlled_trotterized_time_evolution(qc_cU, coeffs, hloc, t, L)

        
        qpe_real, qpe_imag = qc_QPE(L, prepared_state, qc_cU)
        try:
            counts_real = execute(transpile(qpe_real), backend, noise_model=noise_model, shots=shots).result().get_counts()
            phase_est_real = counts_real["0"]/shots - counts_real["1"]/shots
            counts_imag = execute(transpile(qpe_imag), backend, noise_model=noise_model, shots=shots).result().get_counts()
            phase_est_imag = counts_imag["0"]/shots - counts_imag["1"]/shots
            phase_est = phase_est_real + 1j*phase_est_imag
            phase_estimates_with_noise.append((t, phase_est))
            exact_phase = np.exp(-1j*t*eigenvalues_sort[0])
            phase_exacts.append((t, exact_phase))
            #print(phase_est)
        except KeyError:
            continue
    return phase_estimates_with_noise, phase_exacts


def qc_estimate_phases(L, J, g, qc_qetu, eigenvalues_sort, tau, N, shots, depolarizing_error, rqc=True, coeffs=None, qetu_repeat=3):
    backend = qiskit.Aer.get_backend("aer_simulator")
    x1_error = errors.depolarizing_error(depolarizing_error*0.1, 1)
    x2_error = errors.depolarizing_error(depolarizing_error, 2)
    x3_error = errors.depolarizing_error(depolarizing_error*10, 3)
    no_error = errors.depolarizing_error(0, L+1)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(x1_error, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(x2_error, ['cu', 'cx','cy', 'cz'])
    noise_model.add_all_qubit_quantum_error(x3_error, ['ccx'])
    noise_model.add_basis_gates(['meas0'])

    phase_estimates_with_noise = []
    phase_exacts = []
    for n in range(1, N):
        t = n*tau
        qc_cU = qiskit.QuantumCircuit(L+1)

        if rqc:
            V_list = []
            path = f"../../src/rqcopt/results/ising1d_L{L}_t{t}_dynamics_opt_layers5.hdf5"
            try:
                with h5py.File(path, "r") as f:
                    assert f.attrs["L"] == L
                    assert f.attrs["J"] == J
                    assert f.attrs["g"] == g
                    V_list = list(f["Vlist"])
            except FileNotFoundError:
                strang = oc.SplittingMethod.suzuki(2, 1)
                _, coeffs_start_n5 = oc.merge_layers(2*strang.indices, 2*strang.coeffs)
                # divide by 2 since we are taking two steps
                coeffs_start_n5 = [0.5*c for c in coeffs_start_n5]
                print("coeffs_start_n5:", coeffs_start_n5)
                V_list = ising1d_dynamics_opt(5, t, False, coeffs_start_n5, path, niter=16)
            perms = [None if i % 2 == 0 else np.roll(range(L), -1) for i in range(len(V_list))]
            qcs = []
            for V in V_list:
                qc = qiskit.QuantumCircuit(2)
                qc.unitary(V, [0, 1])
                qcs.append(qc)
            for layer, qc_gate in enumerate(qcs):
                cgate = qc_gate.to_gate().control()
                for j in range(L//2):
                    if perms[layer] is not None:
                        qc_cU.append(cgate, [L, L-(perms[layer][2*j]+1), L-(perms[layer][2*j+1]+1)])
                    else:
                        qc_cU.append(cgate, [L, L-(2*j+1), L-(2*j+2)])
        else:
            hloc = construct_ising_local_term(J, g)
            controlled_trotterized_time_evolution(qc_cU, coeffs, hloc, t, L)

        
        qpe_real, qpe_imag = qc_QPE_noisy_sim(L, qc_qetu, qc_cU, qetu_repeat=3)
        try:
            counts_real = execute(transpile(qpe_real), backend, noise_model=noise_model, shots=shots).result().get_counts()
            phase_est_real = counts_real["0"]/shots - counts_real["1"]/shots
            counts_imag = execute(transpile(qpe_imag), backend, noise_model=noise_model, shots=shots).result().get_counts()
            phase_est_imag = counts_imag["0"]/shots - counts_imag["1"]/shots
            phase_est = phase_est_real + 1j*phase_est_imag
            phase_estimates_with_noise.append((t, phase_est))
            exact_phase = np.exp(-1j*t*eigenvalues_sort[0])
            phase_exacts.append((t, exact_phase))
            print(phase_est)
        except KeyError:
            continue
    return phase_estimates_with_noise, phase_exacts



