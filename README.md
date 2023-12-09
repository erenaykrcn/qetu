<h1>Noise Tolerant Ground State Energy Estimation through Quantum Eigenvalue Transformation of Unitary Matrices (QETU)</h1> 

<br>
The work presented in this project delivers an improved error resilience when extracting the eigenvalue information from the ground state prepared through the Quantum Eigenvalue Transformation of Unitary Matrices (QETU) Algorithm, presented by L. Lin et al. [1] Through our approach, we can estimate the eigenvalue up to any arbitary precision under depolarizing probability up to 1e-3 for two qubit gates and up to 1e-4 for single qubit gates. <br>
<br>
<p align="center">
  <img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/adaptive_fuzzy_noise.jpg">

  <b>Figure 1:</b> Absolute Error rate of the estimated ground state energy plotted against the time step required in each time evolution block, under different noise levels. Transverse Field Ising Hamiltonian (TFIM) with parameters: L=6, J=1, h=0, g=1 is used.
</p>

<br>Increasing the desired precision, requires larger time steps in the time evolution blocks. In order to prevent the circuit depth from scaling exponentially with respect to target precision, we use the Riemannian Quantum Circuit Optimization (RQC-Opt), presented by A. Kotil et al. [2].


<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/layers_vs_timestep.jpg">
<br><b>Figure 2:</b> Number of two qubit gate layers used to encode the time evolution block for different time steps. RQC-Opt Algorithm is used to reduce the number of layers.
</p>

<br>The adaptive fuzzy bisection search is implemented through estimating one digit after the floating point by the end of each search. The following linear transformation is applied in the beginning of each search:


<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/eig_trafo0.jpg"  width="40%">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/eig_trafo1.jpg"  width="55%">
<br><b>Figure 3:</b> Linear Transformation of the Hamiltonian, implemented in the Quantum Circuit through changing the time step of the Time Evolution Block (for c1) and applying phase gates (for c2). "d" represents the target digit (d<0 for digits after the floating point) of the search and λ_LowerBound is the lowest value the target eigenvalue can take, given the error margin of the current search stage.
</p>

Lower bound of the eigenvalue is dependent on the outcome of the previous search as follows:


<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/lower_bound_def.jpg"  width="30%">
<br><b>Figure 4:</b> Definition of the eigenvalue lower bound of the current search stage. It is dependant on the estimation result of the previous stage.
</p>

<br>Eigenvalue transformation applied by the QETU circuit, maps the difference between the exact eigenvalue and the lower bound ("magnified" for the current digit) to a cosine function, represented by function "a". On average, our method of adaptively updating (c1, c2) for each search depending on the results of the previous search; maps the target a value to between [-0.5, 0,5], where the slope of cos(pi * x/2) is sharper, hence large approximation error in a, corresponds to smaller approximation errors in lambda.

<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/cos1.png">
<br><b>Figure 5:</b> Cosine tranformation of the previously transformed eigenvalues, through the adaption of (c1, c2). Orange line represents the tanget line to the transformation curve at x = 1. Red points represent the target x values of each search stage and corresponding target digits of each stage. 
<br>
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/cos2.png">
<br><b>Figure 6:</b> Example target x value (set to 0.75) and corresponding error margins in "x" and "a" space, for a succesful search. 
</p>
