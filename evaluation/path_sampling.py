import numpy as np

# Sample paths from MDP under a given policy.
def sample_paths(mdp, policy, max_t, n_paths=10):
    paths = []

    for _ in range(n_paths):
        path = np.zeros((max_t, 4), dtype=int)
        current_state = mdp.init_state

        for time_idx in range(max_t):
            if policy.shape == (mdp.n_states, ):
                action = policy[current_state]
            else:
                action = np.random.choice(mdp.n_actions, size=1, p=policy[current_state])[0]
                
            next_state = np.random.choice(mdp.n_states, size=1, p=mdp.transition_matrix[current_state, action, :])[0]
            reward = mdp.rewards[current_state, action]
            path[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        paths.append(path)

    return paths
