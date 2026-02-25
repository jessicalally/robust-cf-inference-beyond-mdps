from collections import deque
from .mdp import MDP
import numpy as np
from pathlib import Path

class AircraftMDP(MDP):
    def __init__(self):
        self.init_state = 0
        self.n_states = 303
        self.n_actions = 3
        self.states = range(self.n_states)
        self.actions = range(self.n_actions)
        self.transition_matrix = np.load("data/aircraft.npy")
        self.rewards = np.zeros((len(self.states), len(self.actions)))
        self.optimal_policy = np.zeros(len(self.states), dtype=int)
        self.values = np.zeros(len(self.states))

        self.suboptimal_trajectory = np.array([
            [   0,    9,    1,   10],
            [   9,   20,    0,    7],
            [  20,   43,    1,    5],
            [  43,   65,    2,    2],
            [  65,   65,    1, -100],
            [  65,   65,    1, -100],
            [  65,   65,    0, -100],
            [  65,   65,    0, -100],
            [  65,   65,    0, -100],
            [  65,   65,    0, -100],
            [  65,   65,    0, -100],
            [  65,   65,    1, -100]
        ], dtype=int)

        self.optimal_trajectory = np.array([
            [  0,   7,   1,  10],
            [  7,  22,   1,   7],
            [ 22,  46,   1,   5],
            [ 46,  81,   1,   5],
            [ 81, 111,   2,   5],
            [111, 147,   0,   0],
            [147, 183,   0,   0],
            [183, 219,   0,   0],
            [219, 255,   0,   0],
            [255, 291,   0,   0],
            [291, 291,   0, 100],
            [291, 291,   0, 100]
        ], dtype=int)

        # Hyperparameters
        self.discount = 0.9

        # Need to make goal and avoid states terminal, since the original PRISM experiment is
        # a reach-avoid problem (so the transitions aren't terminal.)

        goal_states = [267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302]
        avoid_states = [129, 130, 131, 136, 137, 138, 143, 144, 145, 150, 151, 152, 157, 158, 51, 52, 57, 58, 59, 64, 65, 66, 71, 72, 73, 78, 79, 80, 85, 86, 87, 88, 93, 94, 95, 100, 101, 102, 107, 108, 109, 114, 115, 116, 121, 122, 123, 124]

        for goal_state in goal_states:
            self.transition_matrix[goal_state, :] = np.eye(self.n_states)[goal_state]

        for avoid_state in avoid_states:
            self.transition_matrix[avoid_state, :] = np.eye(self.n_states)[avoid_state]

        # We scale rewards based on how close the state is to the nearest collision state.
        distances = self.compute_state_distances(self.transition_matrix, avoid_states)
        max_dist = np.max(distances[np.isfinite(distances)])

        self.rewards = np.zeros((self.n_states, self.n_actions))
        self.state_rewards = np.zeros(self.n_states)

        for s, d in enumerate(distances):
            if s in goal_states:
                self.rewards[s, :] = 100.0
                self.state_rewards[s] = 100.0
            elif s in avoid_states:
                self.rewards[s, :] = -100.0
                self.state_rewards[s] = -100.0
            elif np.isfinite(d):
                self.rewards[s, :] = int((d / max_dist) * 10)
                self.state_rewards[s] = self.rewards[s, 0]
            else:
                self.rewards[s, :] = 0.0  # unreachable
                self.state_rewards[s] = 0.0

        self.valid_actions = self._extract_valid_actions()
        self.sink_state = np.full(self.n_states, False)

        for s in goal_states:
            self.sink_state[s] = True

        for s in avoid_states:
            self.sink_state[s] = True

        self.max_reward = np.max(self.state_rewards)
        self.env_name = "aircraft_tiny"
        self.value_iteration()


    def _extract_valid_actions(self):
        return np.full(shape=(self.n_states, self.n_actions), fill_value=True)

    
    def compute_state_distances(self, P, avoid_states):
        n_states, n_actions, _ = P.shape
        distances = np.full(n_states, np.inf)
        for g in avoid_states:
            distances[g] = 0

        queue = deque(avoid_states)
        while queue:
            s = queue.popleft()
            for a in range(n_actions):
                predecessors = np.where(P[:, a, s] > 0)[0]  # states that can reach s
                for p in predecessors:
                    if distances[p] == np.inf:
                        distances[p] = distances[s] + 1
                        queue.append(p)

        return distances


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
    def sample_random_trajectory(self, n_steps=12, randomization=1.0):
        suboptimal_policy = self.policy_with_randomization(self.optimal_policy, randomization)

        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = self.init_state

        for time_idx in range(n_steps):
            action = np.random.choice(3, size=1, p=suboptimal_policy[current_state])[0]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0] 
            reward = self.rewards[current_state, action]
            trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        return trajectory.astype(int)
    

    # Samples trajectory produced by optimal policy.
    def sample_optimal_trajectory(self, n_steps=12):
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
    
    def sample_suboptimal_trajectory(self, n_steps=12):
        return self.suboptimal_trajectory
