import numpy as np

class MDP:
    def __init__(self):
        self.init_state = 0
        self.states = range(16)
        self.actions = range(4)
        self.n_states = 16
        self.n_actions = 4
        self.transition_matrix = np.zeros((len(self.states), len(self.actions), len(self.states)))
        self.rewards = np.zeros((len(self.states), len(self.actions)))
        self.optimal_policy = np.zeros(len(self.states), dtype=int) # array with zeros 
        self.values = np.zeros(len(self.states))

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
                
                # Absorbing goal state
                elif state == 15:
                    self.transition_matrix[state, action, :] = 0.0
                    self.transition_matrix[state, action, state] = 1.0
                    self.rewards[state, action] = 100

                elif state in [1, 4, 5]:
                    self.rewards[state, action] = 1.0
                elif state in [2, 8, 9]:
                    self.rewards[state, action] = 2.0
                elif state in [3, 10, 12]:
                    self.rewards[state, action] = 3.0
                elif state in [7, 13]:
                    self.rewards[state, action] = 4.0
                elif state in [11, 14]:
                    self.rewards[state, action] = 5.0 
        
        s_transition_probabilities = np.zeros((len(self.states), len(self.actions), len(self.states)));
        p_r = 0.9

        for action in self.actions:
            other_actions = [a for a in self.actions if a != action]

            for a in other_actions:
                s_transition_probabilities[:, action, :] += (1 - p_r)/3 * self.transition_matrix[:, a, :]

            s_transition_probabilities[:, action, :] += p_r * self.transition_matrix[:, action, :]
        
        assert np.allclose(np.sum(s_transition_probabilities, axis=2), 1.0)
        self.transition_matrix = s_transition_probabilities


    def sample_next_state(self, s, a):
        return np.random.choice(self.n_states, p=self.transition_matrix[s, a])


    def value_iteration(self):
        self.values = np.zeros(len(self.states))

        while True:
            new_values = np.zeros(len(self.states))
            for state in self.states:
                values = np.zeros(len(self.actions))
                for action in self.actions:
                    for next_state in self.states:
                        values[action] += self.transition_matrix[state, action, next_state] * (self.rewards[state, action] + self.discount * self.values[next_state])
                
                new_values[state] = np.max(values)
                self.optimal_policy[state] = np.argmax(values)

            if np.sum(np.abs(new_values - self.values)) < 1e-4:
                break

            self.values = new_values
            
            
    def policy_with_randomization(self, policy, randomization_probability):
        policy_matrix = self.translating_policy_to_matrix(policy)
        random_policy = randomization_probability*np.ones((len(self.states), len(self.actions)))
        random_policy = random_policy + policy_matrix
        random_policy = random_policy / (randomization_probability*len(self.actions) + 1)
        assert np.allclose(np.sum(random_policy, axis=1), 1)
        return random_policy


    def translating_policy_to_matrix(self, policy):
        policy_matrix = np.zeros((len(self.states), len(self.actions)))
        for i in range(len(policy)):
            policy_matrix[i][policy[i]] = 1
        return policy_matrix
    

    # Samples a random trajectory from a suboptimal policy.
    def sample_random_trajectory(self, n_steps=10, randomization=1.0):
        suboptimal_policy = self.policy_with_randomization(self.optimal_policy, randomization)

        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = self.init_state

        for time_idx in range(n_steps):
            action = np.random.choice(4, size=1, p=suboptimal_policy[current_state])[0]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0] 
            reward = self.rewards[current_state, action]
            trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        return trajectory.astype(int)
    

    # Samples trajectory produced by optimal policy.
    def sample_optimal_trajectory(self, n_steps=10):
        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = self.init_state

        for time_idx in range(n_steps): 
            action = self.optimal_policy[current_state]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0] 
            reward = self.rewards[current_state, action]
            trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        return trajectory
    
    # Returns an example suboptimal trajectory that enters the dangerous, terminal state.
    def sample_suboptimal_trajectory(self, n_steps=10):
        return [[0, 1, 1, 1],  [1, 2, 1, 2], [2, 6, 2, -100],  [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100], [6, 6, 0, -100]]
