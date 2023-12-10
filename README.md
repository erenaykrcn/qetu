<h1>Noise Tolerant Ground State Energy Estimation through Quantum Eigenvalue Transformation of Unitary Matrices (QETU)</h1> 

<br>
The work presented in this project delivers an improved error resilience when extracting the eigenvalue information from the ground state prepared through the Quantum Eigenvalue Transformation of Unitary Matrices (QETU) Algorithm, presented by L. Lin et al. [1] Through our approach, we can estimate the eigenvalue up to any arbitary precision under depolarizing probability up to 1e-3 for two qubit gates and up to 1e-4 for single qubit gates. <br>
<br>
<p align="center">
  <img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/adaptive_fuzzy_noise.png">

  <b>Figure 1:</b> Absolute Error rate of the estimated ground state energy plotted against the time step required in each time evolution block, under different noise levels. Transverse Field Ising Hamiltonian (TFIM) with parameters: L=6, J=1, h=0, g=1 is used.
</p>

<br>Increasing the desired precision, requires larger time steps in the time evolution blocks. In order to prevent the circuit depth from scaling exponentially with respect to target precision, we use the Riemannian Quantum Circuit Optimization (RQC-Opt), presented by A. Kotil et al. [2].


<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/time_ev.png">
<br><b>Figure 2:</b> Demonstration of how we encoded the time evolution block in the quantum circuits. On the left, the number of two qubit gate layers per time evolution block with respect to the total time step is given. For smaller total time steps, we do not divide it up to smaller bits (nsteps=1) and optimize the block with RQC-Opt for the total time step. For larger time steps, we divide the total time and optimize the circuit for a smaller dt = (Total Time Step) / nsteps. On the right, we see the optimization results of RQC-Opt, ran for dt values with different orders of magnitude. We observe that by increasing the total number of layers up to 11, we can approximate large time steps with high precision.
</p>

<br>The adaptive fuzzy bisection search is implemented through estimating one digit after the floating point by the end of each search. The following linear transformation is applied in the beginning of each search:

<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/eig_trafo0.jpg"  width="40%">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/eig_trafo1.jpg"  width="48%">
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
<br><br>
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/cos2.png">
<br><b>Figure 6:</b> Example target x value (set to 0.75) and corresponding error margins in "x" and "a" space, for a succesful search. 
</p>

<br>

<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/adaptive_search.png">
<br><b>Figure 7:</b> Demonstration of how each search, estimating the exact "a" value with around 1e-2 precision, can correctly identify the target digit. We see a trade-off between absolute error and the time step required. 
</p>

<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/exp_search.png">
<br><b>Figure 8:</b> An example search, conducted for target precision d=-3. Resulting estimate is a=0.5765, delivering an absolute error of: 1.547e-05 
</p>

<hr>

<h2>Ground State Preparation</h2>

Ground state is prepared through combining the Lindbladian evolution [3] and QETU Circuits [1]. Outcome of short Lindbladian simulation delivers us a significantly large initial overlap, that is then used as the initial state of QETU to amplify the state fidelity to the ground state. 

<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/lind_circuit.png" width="65%">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/lind.png" width="30%">
<br><b>Figure 9:</b> Linbladian Evolution. Diagram on the right demonstrates the convergence of the initial state, whose overlap with the ground state is numerically zero and the end state's overlap is recorded as 0.773 for 1500 steps with each time step set to 1. Circuit on the left, shows a potential implementation of the algorithm.
</p>

<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/qetu.png">
<br><b>Figure 10:</b> QETU Circuit, used to amplify the state fidelity of the output of the short Lindbladian evolution. 
</p>

QETU Algorithm is based on a symmetric Quantum Signal Processing circuit, where the target polynomial is an even step function. After applying this circuit to a given input state; the overlap of the prepared state to the eigenstates, whose eigenenergies correspond to "a" values greater than mu, are amplified. We make use of this circuit to amplify only the overlap with the ground state. To achieve this, we first apply a linear transformation (c1, c2), in order to fit the whole spectrum between [0, pi]. This way we make sure that the cosine transformation a = cos((c1*lambda + c2)/2) is bijective and increasing eigenvalues are mapped to monotonously decreasing "a" values between [0, 1]. 

<br> The cut-off value of the step function (mu) has to be guessed. Ideally, the mu value cuts the ground state energy and the first excited state energy directly in the middle. However, if our guess for mu is poor, we can compensate it by repeating the QETU circuit more times. This due to the monotonously increasing nature of the polynomial and the monotonously decreasing nature of the spectrum in "a" space due to the transformations applied above. 

<p align="center">
<img src="https://github.com/erenaykrcn/qetu/blob/main/theory/figures/poly_plot.jpg">
<br><b>Figure 11:</b> Example target polynomial (degree 30), approximated through convex optimization with Chebyshev polynomials. After determining the target polynomial, we optimize phases for the QETU circuit. Similar approach is employed during fuzzy bisection search. The value of the polynomial at the exact "a" value (for the given (c1, c2)), norm squared, gives us the probability of measuring |0> at the ancilla qubit.
</p>

<br><br><br>

<h3>References</h3>
[1] Yulong Dong, Lin Lin, and Yu Tong PRX Quantum 3, 040305 <br>
[2] Ayse Kotil, Rahul Banerjee, Qunsheng Huang, Christian B. Mendl, Riemannian quantum circuit optimization for Hamiltonian simulation (arXiv:2212.07556) <br>
[3] Z. Ding, C.-F. Chen, L. Lin, Single-ancilla ground state preparation via Lindbladians <a href="https://arxiv.org/abs/2308.15676">arxiv.org/abs/2308.15676</a>
