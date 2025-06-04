from simulator import Simulator
import numpy as np

# Generates IMDP from observed data.

def normalise_imdp(imdp_transition_matrix, observed_path):
    # Normalises the IMDP to meet Assumptions 1 and 2.

    # Assumption 1: the LB of any transition must be at least 1 - UBs of all other transitions, and vice versa.
    for s in range(imdp_transition_matrix.shape[0]):
        for a in range(imdp_transition_matrix.shape[1]):
            lb_sum = np.sum(imdp_transition_matrix[s, a, :, 0])
            ub_sum = np.sum(imdp_transition_matrix[s, a, :, 1])

            for s_prime in range(imdp_transition_matrix.shape[2]):
                # Skip transitions with UB = 0
                if imdp_transition_matrix[s, a, s_prime, 1] > 0:
                    imdp_transition_matrix[s, a, s_prime, 0] = max(imdp_transition_matrix[s, a, s_prime, 0], 1 - (ub_sum - imdp_transition_matrix[s, a, s_prime, 1]))
                    imdp_transition_matrix[s, a, s_prime, 1] = min(imdp_transition_matrix[s, a, s_prime, 1], 1 - (lb_sum - imdp_transition_matrix[s, a, s_prime, 0]))

    # Assumption 2: the LB of the observed transition must be greater than 0.
    epsilon = 0.01 # Minimum value of observed transition.

    transitions = [(observed_path[i][0], observed_path[i][2], observed_path[i][1]) for i in range(len(observed_path) - 1)]

    for (observed_state, observed_action, observed_next_state) in transitions:
        imdp_transition_matrix[observed_state, observed_action, observed_next_state, 0] = max(epsilon, imdp_transition_matrix[observed_state, observed_action, observed_next_state, 0])

    return imdp_transition_matrix


def learn_imdp(mdp, num_episodes=10000, max_steps=20, delta=0.05):
    # Generates data-driven IMDP.
    simulator = Simulator(mdp, num_episodes, max_steps, delta)

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