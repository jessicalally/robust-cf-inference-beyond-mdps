from .mdp import MDP
import numpy as np

class GridWorldMDP(MDP):
    def __init__(self, p_r=0.9):
        self.env_name = f"GridWorld_{p_r}"
        self.init_state = 0
        self.states = range(16)
        self.actions = range(4)
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.sink_state = np.full(self.n_states, False)
        self.state_rewards = np.zeros(self.n_states)
        self.rewards = np.zeros((self.n_states, self.n_actions))
        self.optimal_policy = np.zeros(self.n_states, dtype=int) # array with zeros 
        self.values = np.zeros(self.n_states)

        # Hyperparameters
        self.discount = 0.9

        for state in self.states:
            for action in self.actions:
                next_state = state

                if action == 0:  # up
                    if state not in [0, 1, 2, 3]:
                        next_state = state - 4
                elif action == 1:  # right
                    if state not in [3, 7, 11, 15]:
                        next_state = state + 1
                elif action == 2:  # down
                    if state not in [12, 13, 14, 15]:
                        next_state = state + 4
                elif action == 3:  # left
                    if state not in [0, 4, 8, 12]:
                        next_state = state - 1

                # Set the transition probabilities and rewards
                self.transition_matrix[state, action, next_state] = 1

                # Unsafe absorbing state
                if state in [6]:
                    self.transition_matrix[state, action, :] = 0.0
                    self.transition_matrix[state, action, state] = 1.0
                    self.rewards[state, action] = -100
                    self.state_rewards[state] = -100
                    self.sink_state[state] = True
                
                # Absorbing goal state
                elif state == 15:
                    self.transition_matrix[state, action, :] = 0.0
                    self.transition_matrix[state, action, state] = 1.0
                    self.rewards[state, action] = 100
                    self.state_rewards[state] = 100
                    self.sink_state[state] = True

                elif state in [1, 4, 5]:
                    self.rewards[state, action] = 1.0
                    self.state_rewards[state] = 1.0

                elif state in [2, 8, 9]:
                    self.rewards[state, action] = 2.0
                    self.state_rewards[state] = 2.0

                elif state in [3, 10, 12]:
                    self.rewards[state, action] = 3.0
                    self.state_rewards[state] = 3.0

                elif state in [7, 13]:
                    self.rewards[state, action] = 4.0
                    self.state_rewards[state] = 4.0

                elif state in [11, 14]:
                    self.rewards[state, action] = 5.0
                    self.state_rewards[state] = 5.0

        
        s_transition_probabilities = np.zeros((self.n_states, self.n_actions, self.n_states));

        for action in self.actions:
            other_actions = [a for a in self.actions if a != action]

            for a in other_actions:
                s_transition_probabilities[:, action, :] += (1 - p_r)/3 * self.transition_matrix[:, a, :]

            s_transition_probabilities[:, action, :] += p_r * self.transition_matrix[:, action, :]
        
        assert np.allclose(np.sum(s_transition_probabilities, axis=2), 1.0)
        self.transition_matrix = s_transition_probabilities

        self.valid_actions = self._extract_valid_actions()
        self.value_iteration()
        self.suboptimal_trajectory = self.sample_suboptimal_trajectory()
        self.optimal_trajectory = self.sample_optimal_trajectory()
        self.max_reward = np.max(self.state_rewards)

    
    # Returns an example suboptimal trajectory that enters the dangerous, terminal state.
    def sample_suboptimal_trajectory(self):
        return np.array([[0, 1, 1, 1],  [1, 2, 1, 2], [2, 6, 2, -100],  [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100]])
    
    
    def _extract_valid_actions(self):
        return np.full(shape=(self.n_states, self.n_actions), fill_value=True)
