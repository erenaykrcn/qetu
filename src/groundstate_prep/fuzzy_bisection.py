from ground_state_prep import get_success_prob
import numpy as np


def fuzzy_bisection(ground_state, l, r, d, tolerence, i, hamil, c1, c2, a_max, max_iter = 15):
    x = (r+l)/2
    print("------------------\nx: " + str(x))
    print("d: ", d)
    print("left: ", l)
    print("right: ", r)
    if np.abs((r+l)/2 - a_max) < tolerence or i>max_iter:
        print("End of Search! \n Error: " + str(np.abs((r+l)/2 - a_max)))
        return ((r+l)/2)
    
    # TODO: Determine d depending on the interval length.
    A, state, poly, phis, layers = get_success_prob(ground_state, x, d, 0.99, 10, 1, hamil,  tau=c1, shift=c2, a_max=a_max)
    print("F(a_max)**2: ", poly(a_max)**2)
    
    # TODO: Determine h!
    h = 0.01
    print("Success Prob: ", A)
    
    if A > 0.6:
        return fuzzy_bisection(ground_state, (r+l)/2 - h, r, d, tolerence, i+1, hamil, c1, c2, a_max,)
    elif A < 0.4:
        return fuzzy_bisection(ground_state, l, (r+l)/2 + h, d, tolerence, i+1, hamil, c1, c2, a_max,)
    else:
        print("Not steep enough!")    
        d = d + 4
        return fuzzy_bisection(ground_state, l-h, r+h, d, tolerence, i+1, hamil, c1, c2, a_max)