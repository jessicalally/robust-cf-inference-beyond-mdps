from gridworld import MDP
from gridworld_utils import convert_transition_matrix_to_julia_imdp
from imdp_cf_bounds import CFBoundCalculator
from simulator import Simulator
import numpy as np
import utils

# Generates IMDP from observed data.

def normalise_imdp(imdp_transition_matrix, observed_state, observed_action, observed_next_state):
    # Normalises the IMDP to meet Assumptions 1 and 2.

    # TODO: Assumption 1

    # Assumption 2: the LB of the observed transition must be greater than 0.
    epsilon = 0.01 # Minimum value of observed transition.
    
    imdp_transition_matrix[observed_state, observed_action, observed_next_state, 0] = max(epsilon, imdp_transition_matrix[observed_state, observed_action, observed_next_state, 0])
    return imdp_transition_matrix

def learn_imdp():
    # Generates data-driven IMDP.
    mdp = MDP()
    simulator = Simulator(mdp)

    return simulator.learn_imdp()

def example_toy_imdp():
    # Generates a toy example of an IMDP.
    num_states = 3
    num_actions = 1

    imdp_transition_matrix = np.zeros((num_states, num_actions, num_states, 2))

    np.random.seed(0)

    for s in range(num_states):
        for a in range(num_actions):
            # Generate upper bounds that sum to 1
            upper_bounds = np.random.dirichlet(np.ones(num_states))
            # Generate lower bounds: some fraction of the upper bounds
            lower_bounds = upper_bounds * np.random.uniform(0.3, 0.8, size=num_states)
            
            imdp_transition_matrix[s, a, :, 0] = lower_bounds
            imdp_transition_matrix[s, a, :, 1] = upper_bounds

    return imdp_transition_matrix

observed_state = 0
observed_action = 2
observed_next_state = 1#4

imdp = learn_imdp()
imdp = normalise_imdp(imdp, observed_state, observed_action, observed_next_state)
print("IMDP:", imdp)

cf_bound_calculator = CFBoundCalculator(observed_state, observed_action, observed_next_state, imdp)
cf_imdp = cf_bound_calculator.calculate_all_bounds()
print(f"CFIMDP: {cf_imdp}")

julia_cf_imdp = utils.format_transition_matrix_for_julia(np.array([cf_imdp]), 1, 16, 4)
convert_transition_matrix_to_julia_imdp(julia_cf_imdp)