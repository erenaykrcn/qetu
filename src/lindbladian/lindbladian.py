import numpy as np
from numpy import linalg as LA
import scipy
from utils_lind import crandn, filter_function, construct_shoveling_lindblad_operator

rng = np.random.default_rng(42)


def ideal_lindbladian(hamil, L, tau=1, nsteps = 500):
	eigenvalues, eigenvectors = LA.eig(hamil)
	idx = eigenvalues.argsort()
	eigenvalues_sort = eigenvalues[idx]
	Sw = np.linalg.norm(hamil, ord=2)
	gap = eigenvalues_sort[1] - eigenvalues_sort[0]

	fhat = lambda w: filter_function(w, 2.5*Sw, 0.5*Sw, gap, gap)

	A = np.kron(crandn((2, 2), rng), np.identity(2**(L-1)))
	K = construct_shoveling_lindblad_operator(A, hamil, fhat)

	superoperator_H = -1j * (np.kron(hamil, np.identity(2**L)) - np.kron(np.identity(2**L), hamil.T))
	superoperator_K = np.kron(K, K.conj()) - 0.5 * (np.kron(K.conj().T @ K, np.identity(2**L)) +  np.kron(np.identity(2**L), K.T@K.conj()) )
	expHK = scipy.linalg.expm(tau*(superoperator_H + superoperator_K))
	
	rho_init = crandn((2**L, 2**L), rng)
	rho_init = rho_init @ rho_init.conj().T
	rho_init /= np.trace(rho_init)

	rho = rho_init
	en_list = [np.trace(hamil @ rho).real]
	for n in range(nsteps):
		rho = np.reshape(expHK @ rho.reshape(-1), rho.shape)
		en_list.append(np.trace(hamil @ rho).real)
	return rho, en_list


def circuit_implementation_lindbladian(hamil, L, tau=1, r = 1, nsteps = 1000, A=None):
	eigenvalues, eigenvectors = LA.eig(hamil)
	idx = eigenvalues.argsort()
	eigenvalues_sort = eigenvalues[idx]
	Sw = np.linalg.norm(hamil, ord=2)
	gap = eigenvalues_sort[1] - eigenvalues_sort[0]

	# Filter function in time domain.
	a, da, b, db  = (2.5*Sw, 0.5*Sw, gap, gap)
	erf_f = lambda t: (-1j*np.exp(-np.pi**2*t**2))/(np.pi*t)
	f = lambda t_list: [0.5*(erf_f(t*da)*np.exp(-1j*a*da*t) - erf_f(t*db)*np.exp(-1j*b*db*t)) if t != 0 else 0 for t in t_list]

	# TODO: Why randomized A doesnt work?
	#A = crandn((2, 2), rng)
	#A = (A + A.conj().T)*0.5
	if A is None:
		A = np.array([[-1.24119066+0.j        , -1.83601217+1.00540214j],
	       [-1.83601217-1.00540214j, -1.64034976+0.j        ]])

	Z = np.array([[1, 0], [0, -1]])
	X = np.array([[0, 1], [1, 0]])
	Y = np.array([[0, -1j], [1j, 0]])

	# Discretization.
	S_s = 3
	tau_s = 0.05
	M_s = int(S_s / tau_s)

	# Changing the r changes the convergence behaviour for this specific A!
	A_tilda = []
	w = lambda l: tau_s if np.abs(l) != M_s else tau_s/2
	for l in range(-M_s, M_s+1):
	    s_l = l*tau_s
	    sigma_l = w(l) * (f([s_l])[0].real*X + f([s_l])[0].imag*Y)
	    A_tilda.append(np.kron(scipy.linalg.expm(-0.5j*np.sqrt(tau)/r*(np.kron(sigma_l, A))), np.identity(2**(L-1))))
	    
	U_tau_s_minus = np.kron(np.identity(2), scipy.linalg.expm(-1j*tau_s*hamil))
	U_tau_s_plus  = np.kron(np.identity(2), scipy.linalg.expm(1j*tau_s*hamil))
	U_tau  = np.kron(np.identity(2), scipy.linalg.expm(-1j*tau*hamil))

	Wt = np.identity(2**(L+1))
	for i, At in enumerate(A_tilda):
	    Wt = At @ Wt
	    Wt = U_tau_s_minus @ Wt 
	for At in A_tilda:
	    Wt = At @ Wt
	    Wt = U_tau_s_plus @ Wt

	
	state = np.array([ 1 if i==0 else 0 for i in range(2**(L+1))])
	en_list = [(np.vdot(state[:2**L], hamil@state[:2**L])).real]
	for i in range(nsteps):
	    for j in range(r):
	        state = Wt @ state
	    state = U_tau @ state
	    meas_0 = np.kron(np.array([[1,0],[0,0]]), np.identity(2**L)) @ state 
	    state = meas_0 / np.linalg.norm(meas_0)
	    en_list.append((np.vdot(state[:2**L], hamil@state[:2**L])).real)
	end_state = state[:2**L]
	end_state /= np.linalg.norm(end_state)
	err = (np.vdot(end_state, hamil@end_state) - eigenvalues_sort[0]).real

	return end_state, en_list, err
