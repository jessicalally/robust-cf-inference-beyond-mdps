from .simulator import Simulator
import numpy as np


def learn_imdp(mdp, num_episodes=10000, max_steps=20, gamma=0.05, epsilon=0.01):
    # Generates data-driven IMDP.
    simulator = Simulator(mdp, num_episodes, max_steps, gamma, epsilon, seed=1)

    return simulator.learn_imdp()


def simulate_counts(mdp, prev_state_action_counts, prev_transition_counts, num_episodes=10000, max_steps=20, epsilon=0.01, seed=1, init_state=None):
    simulator = Simulator(mdp, num_episodes, max_steps, epsilon, seed, init_state)
    state_action_counts, transition_counts, example_suboptimal_path = simulator.get_counts(prev_state_action_counts, prev_transition_counts)
    
    return state_action_counts, transition_counts, example_suboptimal_path


def learn_imdp_pac_guaranteed(mdp, state_action_counts, transition_counts, num_episodes=10000, max_steps=20, gamma=0.05, epsilon=0.01, method="hoeffding"):
    simulator = Simulator(mdp, num_episodes, max_steps, epsilon, seed=1)
    return simulator.learn_imdp_pac_guaranteed(state_action_counts, transition_counts, gamma, method=method)


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


