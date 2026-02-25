from abc import ABC, abstractmethod
import numpy as np

class MDP(ABC):

    def __init__(self):
        self.init_state = None
        self.states = None
        self.actions = None
        self.n_states = None
        self.n_actions = None
        self.transition_matrix = None
        self.sink_state = None
        self.rewards = None
        self.optimal_policy = np.zeros(self.n_states, dtype=int)
        self.values = None

        # Hyperparameters
        self.discount = None


    def set_suboptimal_trajectory(self, suboptimal_trajectory):
        self.suboptimal_trajectory = suboptimal_trajectory

    
    def set_optimal_trajectory(self, optimal_trajectory):
        self.optimal_trajectory = optimal_trajectory


    def sample_next_state(self, s, a):
        return self.sink_state[s], np.random.choice(self.n_states, p=self.transition_matrix[s, a])

    # Value iteration method that takes into account valid actions.
    def value_iteration(self):
        self.values = np.zeros(self.n_states)

        idx=0

        while True:
            print(f"iter={idx}")
            new_values = np.zeros(self.n_states)
            
            for state in self.states:
                best_value = -np.inf
                best_action = -1

                for action in self.actions:
                    if not self.valid_actions[state, action]:
                        continue

                    q_val = 0.0

                    for next_state in self.states:                        
                        q_val += self.transition_matrix[state, action, next_state] * (
                            self.rewards[state, action] + self.discount * self.values[next_state]
                        )

                    if q_val > best_value:
                        best_value = q_val
                        best_action = action

                new_values[state] = best_value
                self.optimal_policy[state] = best_action

            print(f"Delta: {np.sum(np.abs(new_values - self.values))}")

            if np.sum(np.abs(new_values - self.values)) < 1e-4:
                break

            self.values = new_values
            idx += 1

        print(f"RESULT={self.values[self.init_state]}")
        print(f"POLICY={self.optimal_policy}")


    # Generates a policy with p=randomization_probability of choosing a random action.
    def policy_with_randomization(self, policy, randomization_probability):
        # Convert deterministic policy into one-hot matrix
        policy_matrix = self.translating_policy_to_matrix(policy)

        num_states, num_actions = self.valid_actions.shape
        random_policy = np.zeros((num_states, num_actions))

        for s in range(num_states):
            valid = self.valid_actions[s]
            valid_actions = np.where(valid)[0]

            if len(valid_actions) == 0:
                # no valid actions? leave row as zeros (shouldn't happen in a proper MDP)
                continue

            # uniform random distribution over valid actions
            uniform_probs = np.zeros(num_actions)
            uniform_probs[valid_actions] = 1.0 / len(valid_actions)

            # mix deterministic policy with randomization
            random_policy[s] = (
                (1 - randomization_probability) * policy_matrix[s]
                + randomization_probability * uniform_probs
            )

        # sanity check
        assert np.allclose(np.sum(random_policy, axis=1), 1), "Policy rows must sum to 1"
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
            action = np.random.choice(self.n_actions, size=1, p=suboptimal_policy[current_state])[0]
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
    

    @abstractmethod
    def sample_suboptimal_trajectory(self, n_steps=10):
        pass


    def get_possible_transitions(self):
        # Returns list of transitions that are possible - we assume knowledge of the tree structure of the MDP.
        possible_transitions = np.nonzero(self.transition_matrix)
        return list(zip(*possible_transitions))
    

    def get_possible_nondeterministic_transitions(self):
        # Returns list of transitions that are possible and nondeterministic - we assume knowledge of the tree structure of the MDP.

        # Get the shape of the transition matrix
        num_states, num_actions, _ = self.transition_matrix.shape
        nondeterministic_transitions = []

        for s in range(num_states):
            for a in range(num_actions):
                next_state_probs = self.transition_matrix[s, a]
                num_next_states = np.count_nonzero(next_state_probs)
                
                if num_next_states > 1:
                    for s_prime in np.nonzero(next_state_probs)[0]:
                        nondeterministic_transitions.append((s, a, s_prime))

        return nondeterministic_transitions
    

class RewardMDP(MDP):
    def __init__(self, env_name, states, actions, init_state, transition_matrix, valid_actions, rewards, state_rewards, max_t, suboptimal_trajectory, optimal_trajectory):
        self.env_name = env_name
        self.init_state = init_state
        self.states = states
        self.actions = list(actions)
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.transition_matrix = transition_matrix
        self.valid_actions = valid_actions
        self.optimal_policy = np.zeros(self.n_states, dtype=int)
        self.values = np.zeros(len(self.states))
        self.state_rewards = state_rewards
        self.rewards = rewards
        self.discount = 1.0
        self.sink_state = self.get_sink_states()
        self.max_t = max_t

        self.value_iteration()

        self.optimal_trajectory = self.sample_optimal_trajectory(self.max_t)
        self.suboptimal_trajectory = self.sample_random_trajectory(self.max_t)


    def get_sink_states(self):
        num_states, num_actions, _ = self.transition_matrix.shape
        eye = np.eye(num_states)
        
        sink_states = np.full(num_states, True)

        for s in range(num_states):
            for a in range(num_actions):
                if not np.all(self.transition_matrix[s, a] == eye[s]):
                    sink_states[s] = False
                    break

        return sink_states


    def sample_suboptimal_trajectory(self, n_steps=10):
        return self.sample_random_trajectory(n_steps)
    

    # Samples a random trajectory from a suboptimal policy.
    def sample_random_trajectory(self, n_steps=10, randomization=1.0):
        suboptimal_policy = self.policy_with_randomization(self.optimal_policy, randomization)

        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = self.init_state

        for time_idx in range(n_steps):
            action = np.random.choice(self.n_actions, size=1, p=suboptimal_policy[current_state])[0]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0] 
            reward = self.rewards[current_state, action]
            trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        return trajectory.astype(int)
    
    


class Model_Checking_MDP:
    def __init__(self, env_name, states, actions, init_state, transition_matrix, optimal_policy, valid_actions, goal_states, avoid_states, suboptimal_trajectory, optimal_trajectory):
        self.env_name = env_name
        self.init_state = init_state
        self.states = states
        self.actions = list(actions)
        self.n_states = len(states)
        self.n_actions = len(actions)
        self.transition_matrix = transition_matrix
        self.valid_actions = valid_actions
        self.optimal_policy = optimal_policy
        self.values = np.zeros(len(self.states))

        if avoid_states is not None:
            self.collision_states = set(avoid_states)
        else:
            self.collision_states = set()

        self.goal_states = set(goal_states)
        self.optimal_trajectory = self.sample_optimal_trajectory()
        self.suboptimal_trajectory = self.sample_random_trajectory()


    def sample_next_state(self, s, a):
        s_prime = np.random.choice(self.n_states, p=self.transition_matrix[s, a])

        if s_prime in self.goal_states or s_prime in self.collision_states:
            return True, s_prime

        return False, s_prime
            

    def policy_with_randomization(self, policy, randomization_probability):
        policy_matrix = self.translating_policy_to_matrix(policy)
        random_policy = randomization_probability*np.ones((len(self.states), len(self.actions)))
        random_policy = random_policy + policy_matrix
        random_policy = random_policy / (randomization_probability*len(self.actions) + 1)
        assert np.allclose(np.sum(random_policy, axis=1), 1)
        return random_policy


    def translating_policy_to_matrix(self, policy):
        policy_matrix = np.zeros((len(self.states), len(self.actions)))

        for i in self.states:
            policy_matrix[i][int(policy[i])] = 1

        return policy_matrix
    

    def set_suboptimal_trajectory(self, suboptimal_trajectory):
        self.suboptimal_trajectory = suboptimal_trajectory

    
    def set_optimal_trajectory(self, optimal_trajectory):
        self.optimal_trajectory = optimal_trajectory


    # Samples a random trajectory from a suboptimal policy.
    def sample_random_trajectory(self, n_steps=10, randomization=1.0):
        suboptimal_policy = self.policy_with_randomization(self.optimal_policy, randomization)

        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = self.init_state

        for time_idx in range(n_steps):
            action = np.random.choice(self.n_actions, size=1, p=suboptimal_policy[current_state])[0]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0]

            if next_state in self.collision_states:
                trajectory[time_idx, :] = np.array([current_state, next_state, action, -1])
            elif next_state in self.goal_states:
                trajectory[time_idx, :] = np.array([current_state, next_state, action, 1])
            else:
                trajectory[time_idx, :] = np.array([current_state, next_state, action, 0])

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
            
            if next_state in self.collision_states:
                trajectory[time_idx, :] = np.array([current_state, next_state, action, -1])
            elif next_state in self.goal_states:
                trajectory[time_idx, :] = np.array([current_state, next_state, action, 1])
            else:
                trajectory[time_idx, :] = np.array([current_state, next_state, action, 0])

            current_state = next_state

        print(f"Optimal: {trajectory}")

        return trajectory
    

    # Returns an example suboptimal trajectory that enters the dangerous, terminal state.
    def sample_suboptimal_trajectory(self, n_steps=10):
        return self.suboptimal_trajectory
    

    def get_possible_transitions(self):
        # Returns list of transitions that are possible - we assume knowledge of the tree structure of the MDP.
        possible_transitions = np.nonzero(self.transition_matrix)
        return list(zip(*possible_transitions))
    

    def get_possible_nondeterministic_transitions(self):
        # Returns list of transitions that are possible and nondeterministic - we assume knowledge of the tree structure of the MDP.

        # Get the shape of the transition matrix
        num_states, num_actions, _ = self.transition_matrix.shape
        nondeterministic_transitions = []

        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_state_probs = self.transition_matrix[s, a]
                num_next_states = np.count_nonzero(next_state_probs)
                
                if num_next_states > 1:
                    for s_prime in np.nonzero(next_state_probs)[0]:
                        nondeterministic_transitions.append((s, a, s_prime))

        return nondeterministic_transitions