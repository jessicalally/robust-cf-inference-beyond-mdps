import h5py
import numpy as np

def format_transition_matrix_for_julia(interval_CF_MDP, n_timesteps, n_states, n_actions):
    # We have to treat each (t, s) as a separate state, and only allow the transitions to the next time step.
    transition_matrix = {}

    for t in range(n_timesteps):
        for s in range(n_states):
            lower_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))
            upper_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))

            for a in range(n_actions):
                for s_prime in range(n_states):
                    bounds = interval_CF_MDP[t][s, a, s_prime]
                    lower_transition_probs[((t+1)*16) + s_prime, a] = bounds[0]
                    upper_transition_probs[((t+1)*16) + s_prime, a] = bounds[1]

            transition_matrix[(t, s)] = (lower_transition_probs, upper_transition_probs)

    # Make last states sink states.
    for s in range(n_states):
        lower_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))
        upper_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))

        lower_transition_probs[(n_timesteps*n_states) + s, :] = 1.0
        upper_transition_probs[(n_timesteps*n_states) + s, :] = 1.0

        transition_matrix[(t+1, s)] = (lower_transition_probs, upper_transition_probs)

    return transition_matrix


def load_generated_policy(filename, n_timesteps, n_states):
    pi = np.zeros(shape=(n_timesteps, n_states))

    with h5py.File(filename, "r") as file:
        data = file["data"]

        for t in range(n_timesteps):
            ref = data[t]
            res = np.array(file[ref])
            int_array = [int(byte_str) for byte_str in res]
            
            for s in range(n_states):
                pi[t, s] = int_array[(t*n_states)+s]
    
    return pi.astype(int)


def load_value_function(filename, n_timesteps, n_states):
    V = np.zeros(shape=(n_timesteps, n_states))

    with h5py.File(filename, "r") as file:
        data = file["data"]

        for t in range(1, n_timesteps+1):
            ref = data[n_timesteps-t]
            res = np.array(file[ref])
            
            float_array = [float(byte_str) for byte_str in res]

            for s in range(n_states):
                V[t-1, s] = float_array[((t-1) * n_states)+s]
    
    return V.astype(float)
