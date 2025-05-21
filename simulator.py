# Simulates transitions from an MDP.

from collections import defaultdict
from math import sqrt, log
import numpy as np
from tqdm import trange

class Simulator:

    def __init__(self, mdp):
        self.mdp = mdp
        self.num_episodes = 10000
        self.max_steps = 20
        self.delta = 0.05
        self.possible_transitions = mdp.get_possible_transitions()


    def _simulate_transitions(self):
        transitions = []

        for _ in trange(self.num_episodes):
            state = np.random.randint(self.mdp.n_states)  # start from random state
            for _ in range(self.max_steps):
                action = np.random.randint(self.mdp.n_actions)  # random policy
                next_state = self.mdp.sample_next_state(state, action)
                transitions.append((state, action, next_state))
                state = next_state

        return transitions
    

    def learn_imdp(self):
        state_action_counts = defaultdict(int)
        transition_counts = defaultdict(int)

        observed_transitions = self._simulate_transitions()

        for s, a, s_next in observed_transitions:
            state_action_counts[(s, a)] += 1
            transition_counts[(s, a, s_next)] += 1

        imdp = np.zeros((self.mdp.n_states, self.mdp.n_actions, self.mdp.n_states, 2))  # [LB, UB]

        for s in range(self.mdp.n_states):
            for a in range(self.mdp.n_actions):
                N = state_action_counts.get((s, a), 0)
                for s_prime in range(self.mdp.n_states):
                    if (s, a, s_prime) in self.possible_transitions:
                        count = transition_counts.get((s, a, s_prime), 0)
                        if N > 0:
                            p_hat = count / N
                            # Hoeffding-style bound
                            epsilon = sqrt(log(2 / self.delta) / (2 * N))
                            lb = max(0.0, p_hat - epsilon)
                            ub = min(1.0, p_hat + epsilon)
                        else:
                            lb, ub = 0.0, 1.0  # No data: full uncertainty

                        imdp[s, a, s_prime, 0] = lb
                        imdp[s, a, s_prime, 1] = ub

        return imdp

