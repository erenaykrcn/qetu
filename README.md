<h1>Noise Tolerant Ground State Energy Estimation through Quantum Eigenvalue Transformation of Unitary Matrices (QETU)</h1> 

<br>
The work presented in this project delivers an improved error resilience to extracting the eigenvalue information from the ground state prepared through Quantum Eigenvalue Transformation of Unitary Matrices (QETU) Algorithm, presented by L. Lin et al. [1] Through our approach, we can estimate the eigenvalue up to any arbitary precision under depolarizing probability up to 1e-3 for the two qubit gates and up to 1e-4 for the single qubit gates. 

<br>
![](https://github.com/erenaykrcn/qetu/blob/main/theory/figures/adaptive_fuzzy_noise.jpg)

<b>Figure 1:</b> Absolute Error rate of the estimated ground state energy plotted against the time step required in each time evolution block, under different noise levels. Transverse Field Ising Hamiltonian (TFIM) with parameters: L=6, J=1, h=0, g=1 is used.


<br>Increasing the desired precision, requires larger time steps in the time evolution blocks. In order to prevent the circuit depth from scaling exponentially with respect to target precision, we use the Riemannian Quantum Circuit Optimization (RQC-Opt), presented by A. Kotil et al. [2].


![](https://github.com/erenaykrcn/qetu/blob/main/theory/figures/layers_vs_timestep.jpg)
<br><b>Figure 2:</b> Number of two qubit gate layers used to encode the time evolution block for different time steps. RQC-Opt Algorithm is used to reduce the number of layers.

 <br>



<br>

