import numpy as np
from scipy.stats import hypergeom

# Constants
MAX_POPULATION = 10

state_space = [(S, I, V) for S in range(MAX_POPULATION + 1)
                         for I in range(MAX_POPULATION + 1)
                         for V in range(2 * MAX_POPULATION + 1)]
num_states = len(state_space)
state_index = {state: i for i, state in enumerate(state_space)}
state_from_index = {i: state for i, state in enumerate(state_space)}

# Action Space
actions = ["NIL", "V_I", "V_S"]
action_index = {"NIL": 0, "V_I": 1, "V_S": 2}
action_from_index = {0: "NIL", 1: "V_I", 2: "V_S"}
num_actions = len(actions)

class MDP:
    def __init__(self, states, state_index, actions, action_index):
        self.states = states
        self.n_states = len(states)
        self.state_index = state_index
        self.actions = actions
        self.n_actions = len(actions)
        self.action_index = action_index
        self.initial_state = self.get_initial_state()
        self.T = 7

        # Compute Transition and Reward Matrices
        transition_matrix = np.zeros((num_actions, num_states, num_states))
        reward_matrix = np.zeros((num_states, num_actions))

        for action_idx, action in enumerate(actions):
            for state in state_space:
                S, I, V = state
                state_idx = state_index[state]
                transitions = self._compute_transitions(S, I, V, action)

                # Update transition matrix
                for next_state, prob in transitions.items():
                    next_state_idx = state_index[next_state]
                    transition_matrix[action_idx, state_idx, next_state_idx] = prob

                # Update reward matrix (negative of the number of infected individuals)
                reward_matrix[state_idx, action_idx] = -I

        self.transition_matrix = np.nan_to_num(transition_matrix)
        self.reward_matrix = reward_matrix

    # Function to compute transition probabilities
    def _compute_transitions(self, S, I, V, action):
        # Initialise Transition Matrix and Reward Matrix
        M = S + I  # Total population for hypergeometric distribution
        transitions = {}

        if action == "NIL":
            N = S
            n = min(S, I)
            V_prime = V

            for k in range(S + 1):
                prob = hypergeom(M, n, N).pmf(k)
                S_prime, I_prime = S - k, I + k

                if S_prime >= 0 and I_prime <= MAX_POPULATION:
                    transitions[(S_prime, I_prime, V_prime)] = prob

        elif action == "V_I" and I > 0 and V > 0:
            M -= 1
            N = S
            n = min(S, I - 1)
            V_prime = V - 1

            for k in range(S + 1):
                prob = hypergeom(M, n, N).pmf(k)
                S_prime, I_prime = S - k, I - 1 + k
                if S_prime >= 0 and I_prime <= MAX_POPULATION:
                    transitions[(S_prime, I_prime, V_prime)] = prob

        elif action == "V_S" and S > 0 and V > 0:
            M -= 1
            N = S - 1
            n = min(S - 1, I)
            V_prime = V - 1
            for k in range(S):
                prob = hypergeom(M, n, N).pmf(k)
                S_prime, I_prime = S - 1 - k, I + k
                if S_prime >= 0 and I_prime <= MAX_POPULATION:
                    transitions[(S_prime, I_prime, V_prime)] = prob

        return transitions

    def get_initial_state(self):
        return (9, 1, 20)


epidemic_mdp = MDP(state_space, state_index, actions, action_index)
