from ground_state_prep import get_success_prob
from ground_state_prep_qiskit import qetu_rqc_oneLayer
from utils_gsp import QETU_cf, U_tau

import numpy as np
import qiskit
from qiskit import execute, transpile, Aer
from qiskit.providers.aer.noise import NoiseModel, errors
from qiskit.circuit.library import StatePreparation

from qiskit.quantum_info import state_fidelity


def fuzzy_bisection(ground_state, l, r, d, tolerence, i, hamil, c1, c2, a_max, max_iter = 15):
    x = (r+l)/2
    print("------------------\nx: " + str(x))
    print("d: ", d)
    print("left: ", l)
    print("right: ", r)
    a_est = (r+l)/2
    
    if np.abs(a_est - a_max) < tolerence or i>max_iter:
        print("End of Search! \n Error: ", a_est - a_max)
        return ((r+l)/2)
    
    # TODO: Determine d depending on the interval length.
    A, state, poly, phis, layers, QETU_cf_mat = get_success_prob(ground_state, x, d, 0.99, 10, 1, hamil,  tau=c1, shift=c2, a_max=a_max)
    print("F(a_max)**2: ", poly(a_max)**2)
    
    # TODO: Determine h!
    h = 0.01
    print("Success Prob: ", A)
    
    if A > 0.6:
        return fuzzy_bisection(ground_state, (r+l)/2 - h, r, d, tolerence, i+1, hamil, c1, c2, a_max)
    elif A < 0.4:
        return fuzzy_bisection(ground_state, l, (r+l)/2 + h, d, tolerence, i+1, hamil, c1, c2, a_max)
    else:
        print("Not steep enough!")    
        d = d + 4
        return fuzzy_bisection(ground_state, l-h, r+h, d, tolerence, i+1, hamil, c1, c2, a_max)


def fuzzy_bisection_noisy(  qc_qetu, L, J, g, l, r, d, tolerence, i, c1, c2, 
                            a_values, depolarizing_error, max_iter = 15, qetu_layers=3, RQC_layers=9, 
                            qetu_initial_state=None, nshots=1e5, ground_state=None, split_U=15, reuse_RQC=0,
			                lower_treshold=0.4, upper_treshold=0.6, qc_U_custom=None,
                            custom_qc_QETU_cf_R=None, qc_cU_custom=None, hamil=None
    ):
    x = (r+l)/2
    a_max = a_values[0]
    print("------------------\nx: " + str(x))
    print("d: ", d)
    print("left: ", l)
    print("right: ", r)
    a_est = (r+l)/2
    c = 0.95
    phis_max_iter = 10
    
    if i>max_iter:
        print("End of Search! \n Error: ", a_est - a_max)
        return ((r+l)/2)
    
    t = c1/2
    last_layer, phis, _ = qetu_rqc_oneLayer(L, J, g, t, x, a_values, d=d, c=c, c2=c2, 
        max_iter_for_phis=phis_max_iter, reuse_RQC=reuse_RQC, RQC_layers=RQC_layers, 
        split_U=split_U, qc_U_custom=qc_U_custom, custom_qc_QETU_cf_R=custom_qc_QETU_cf_R,
        qc_cU_custom=qc_cU_custom, hamil=hamil
        )

    qc = qiskit.QuantumCircuit(L+1, 1)
    backend = Aer.get_backend("statevector_simulator")
    if qetu_initial_state is not None:
        qc.initialize(qetu_initial_state)
    state_prep_gates = []
    for i in range(qetu_layers):
        qc.append(qc_qetu.to_gate(), [i for i in range(L+1)])
        #qc.reset(L)
        bR = execute(transpile(qc), backend).result().get_statevector().data
        aR = np.kron(np.array([[1,0],[0,0]]), np.identity(2**L)) @ bR
        aR = aR / np.linalg.norm(aR)
        statePrep_Gate = StatePreparation(aR, label=f'meas{i}')
        state_prep_gates.append(statePrep_Gate)
        qc.reset([i for i in range(L+1)])
        qc.append(statePrep_Gate, [i for i in range(L+1)])

    if ground_state is not None:
        np_arr = np.array(execute(transpile(qc), backend).result().get_statevector().data)
        print("state_fidelity:", state_fidelity(np_arr[:2**L], ground_state))
    
    qc.append(last_layer.to_gate(), [i for i in range(L+1)])
    qc.measure(L, 0)

    backend = qiskit.Aer.get_backend("aer_simulator")
    x1_error = errors.depolarizing_error(depolarizing_error*0.1, 1)
    x2_error = errors.depolarizing_error(depolarizing_error, 2)
    no_error = errors.depolarizing_error(0, L+1)
    #x3_error = errors.depolarizing_error(depolarizing_error*10, 3)
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(no_error, state_prep_gates)
    noise_model.add_basis_gates(state_prep_gates)
    noise_model.add_all_qubit_quantum_error(x1_error, ['u1', 'u2', 'u3', 'rz', 'sx'])
    noise_model.add_all_qubit_quantum_error(x2_error, ['cu', 'cx','cy', 'cz'])
    #noise_model.add_all_qubit_quantum_error(x3_error, ['ccx', 'ccy','ccz'])
    print(noise_model)

    #print("Layers: ", transpile(qc, basis_gates=noise_model.basis_gates).depth())

    A = execute(transpile(qc), backend, basis_gates=noise_model.basis_gates, 
        noise_model=noise_model, shots=nshots).result().get_counts()["0"]/nshots


    # TODO: Determine h!
    h = 0.01
    print("Success Prob: ", A)
    
    if A > upper_treshold:
        return fuzzy_bisection_noisy(qc_qetu, L, J, g, (r+l)/2 - h, r, d, tolerence, i+1, c1, c2, a_values, 
            depolarizing_error, max_iter, qetu_layers, RQC_layers,  qetu_initial_state, nshots, ground_state, 
            split_U, reuse_RQC=reuse_RQC,qc_U_custom=qc_U_custom,custom_qc_QETU_cf_R=custom_qc_QETU_cf_R, 
            qc_cU_custom=qc_cU_custom,  hamil=hamil)
    elif A < lower_treshold:
        return fuzzy_bisection_noisy(qc_qetu, L, J, g, l, (r+l)/2 + h, d, tolerence, i+1, c1, c2, a_values, 
            depolarizing_error, max_iter, qetu_layers, RQC_layers, qetu_initial_state, nshots, ground_state, 
            split_U, reuse_RQC=reuse_RQC, qc_U_custom=qc_U_custom,custom_qc_QETU_cf_R=custom_qc_QETU_cf_R, 
            qc_cU_custom=qc_cU_custom,  hamil=hamil)
    else:
        print("Not steep enough! Search ended!")    
        d = d + 4
        nshots = 10*nshots if nshots < 1e6 else nshots

        if d > 34:
            return ((r+l)/2)
        else:
            return fuzzy_bisection_noisy(qc_qetu, L, J, g, l-h, r+h, d, tolerence, i+1, c1, c2, a_values, depolarizing_error, 
                max_iter, qetu_layers, RQC_layers, qetu_initial_state, nshots, ground_state, split_U, 
                reuse_RQC=reuse_RQC, qc_U_custom=qc_U_custom, custom_qc_QETU_cf_R=custom_qc_QETU_cf_R, 
                qc_cU_custom=qc_cU_custom,  hamil=hamil)

