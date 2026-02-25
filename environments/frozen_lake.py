import h5py
import numpy as np
from .mdp import MDP

class FrozenLakeMDP(MDP):
    def __init__(self):
        self.init_state = 0
        self.n_states = 17
        self.n_actions = 4
        self.states = range(17)
        self.actions = range(4)
        self.transition_matrix = np.load("data/frozen_lake.npy")
        self.rewards = np.zeros((len(self.states), len(self.actions)))
        self.state_rewards = np.zeros(self.n_states)
        self.optimal_policy = np.zeros(len(self.states), dtype=int)
        self.values = np.zeros(len(self.states))

        # Hyperparameters
        self.discount = 0.9

        for state in self.states:
            for action in self.actions:
                # Unsafe absorbing states
                if state in [5, 7, 11, 12]:
                    self.rewards[state, action] = -100
                # Absorbing goal state
                elif state == 15:
                    self.rewards[state, action] = 100
                elif state in [1, 4]:
                    self.rewards[state, action] = 1.0
                elif state in [2, 8]:
                    self.rewards[state, action] = 2.0
                elif state in [3, 6, 9]:
                    self.rewards[state, action] = 3.0
                elif state in [10, 13]:
                    self.rewards[state, action] = 4.0 
                elif state in [14]:
                    self.rewards[state, action] = 5.0

            self.state_rewards[state] = self.rewards[state, 0]

        self.valid_actions = self._extract_valid_actions()
        self.sink_state = np.full(self.n_states, False)

        for s in [5, 7, 11, 12, 15]:
            self.sink_state[s] = True

        self.max_reward = np.max(self.state_rewards)
        self.env_name = "frozen_lake"
        
        self.value_iteration()

        self.suboptimal_trajectory = self.sample_suboptimal_trajectory()
        self.optimal_trajectory = self.sample_optimal_trajectory()

    def _extract_valid_actions(self):
        return np.full(shape=(self.n_states, self.n_actions), fill_value=True)

    def sample_suboptimal_trajectory(self):
        return np.array([
            [  0,   1,   3,    0],
            [  1,   5,   2,    1],
            [  5,  16,   3, -100],
            [ 16,  16,   1,    0],
            [ 16,  16,   1,    0],
            [ 16,  16,   1,    0],
            [ 16,  16,   3,    0],
            [ 16,  16,   2,    0],
            [ 16,  16,   2,    0],
            [ 16,  16,   0,    0],
            [ 16,  16,   0,    0],
            [ 16,  16,   2,    0],
            [ 16,  16,   1,    0],
            [ 16,  16,   0,    0],
            [ 16,  16,   3,    0],
            [ 16,  16,   2,    0],
            [ 16,  16,   1,    0],
            [ 16,  16,   1,    0],
            [ 16,  16,   2,    0],
            [ 16,  16,   1,    0]
        ]).astype(int)

    
    def sample_optimal_trajectory(self):
        return np.array([
            [  0,   1,   2,   0],
            [  1,   0,   3,   1],
            [  0,   4,   2,   0],
            [  4,   4,   0,   1],
            [  4,   8,   0,   1],
            [  8,   9,   3,   2],
            [  9,   8,   1,   3],
            [  8,   9,   3,   2],
            [  9,  10,   1,   3],
            [ 10,   6,   0,   4],
            [  6,  10,   0,   3],
            [ 10,  14,   0,   4],
            [ 14,  14,   1,   5],
            [ 14,  15,   1,   5],
            [ 15,  16,   0, 100],
            [ 16,  16,   0,   0],
            [ 16,  16,   0,   0],
            [ 16,  16,   0,   0],
            [ 16,  16,   0,   0],
            [ 16,  16,   0,   0]
        ]).astype(int)
        

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
    def sample_random_trajectory(self, n_steps=20, randomization=1.0):
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
    