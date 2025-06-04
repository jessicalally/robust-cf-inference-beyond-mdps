import h5py
import numpy as np
import pickle

def format_transition_matrix_for_julia(interval_CF_MDP, n_timesteps, n_states, n_actions):
    # We have to treat each (t, s) as a separate state, and only allow the transitions to the next time step.
    transition_matrix = {}

    for t in range(n_timesteps):
        for s in range(n_states):
            lower_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))
            upper_transition_probs = np.zeros(shape=(n_states * (n_timesteps+1), n_actions))

            for a in range(n_actions):
                for s_prime in range(n_states):
                    bounds = interval_CF_MDP[t, s, a, s_prime]
                    lower_transition_probs[((t+1)*n_states) + s_prime, a] = bounds[0]
                    upper_transition_probs[((t+1)*n_states) + s_prime, a] = bounds[1]

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


def save_imdp_to_file(filename, imdp):
    n_states, n_actions = imdp.shape[:2]
    
    with open(filename, 'w') as f:
        for s in range(n_states):
            for a in range(n_actions):
                f.write(f"State {s}, Action {a}:\n")
                for s_prime in range(n_states):
                    lb, ub = imdp[s, a, s_prime]
                    if lb > 0 or ub > 0:
                        f.write(f"  -> State {s_prime}: [{lb:.4f}, {ub:.4f}]\n")
                f.write("\n")


def generate_julia_file(filename, interval_mdp_code, transition_probs_code):
    with open(filename, 'w') as file:
        file.write(transition_probs_code)
        file.write(interval_mdp_code)

def create_transition_prob_string(lower_probs, upper_probs):
    global TRANSITION_PROB_COUNTER

    lower_str = "\n        ".join([" ".join(map(str, row)) for row in lower_probs])
    upper_str = "\n        ".join([" ".join(map(str, row)) for row in upper_probs])

    prob_str = f"""
    prob{TRANSITION_PROB_COUNTER} = IntervalProbabilities(;
        lower = [
            {lower_str}
        ],
        upper = [
            {upper_str}
        ],
    )
    """

    TRANSITION_PROB_COUNTER += 1

    return prob_str

def create_transition_prob_string_compressed(lower_probs, upper_probs):
    global TRANSITION_PROB_COUNTER

    lower_str_parts = []
    upper_str_parts = []

    for col in range(lower_probs.shape[1]):
        non_zero_probs_indices = np.nonzero(upper_probs[:, col])[0] # use upper bound indicies because some states might have LB=0

        non_zero_probs_lower = ", ".join(map(str, lower_probs[non_zero_probs_indices, col]))
        non_zero_probs_upper = ", ".join(map(str, upper_probs[non_zero_probs_indices, col]))
        non_zero_probs_indices = [idx+1 for idx in non_zero_probs_indices]
        non_zero_probs_indices = ", ".join(map(str, non_zero_probs_indices))
        
        lower_str_parts.append(f"\n\t\t\tSparseVector({lower_probs.shape[0]}, [{non_zero_probs_indices}], [{non_zero_probs_lower}]),")

        upper_str_parts.append(f"\n\t\t\tSparseVector({upper_probs.shape[0]}, [{non_zero_probs_indices}], [{non_zero_probs_upper}]),")

    lower_str = "".join(lower_str_parts)
    upper_str = "".join(upper_str_parts)

    prob_str = f"""
    prob{TRANSITION_PROB_COUNTER} = IntervalProbabilities(;
        lower = sparse_hcat({lower_str}
        ),
        upper = sparse_hcat({upper_str}
        ),
    )
    """

    TRANSITION_PROB_COUNTER += 1

    return prob_str


def convert_transition_matrix_to_julia_imdp(mdp_name, tra_filename, filename, rewards, initial_state, n_timesteps, n_states, n_actions):
    global TRANSITION_PROB_COUNTER
    TRANSITION_PROB_COUNTER = 1

    with open(tra_filename, "rb") as f:
        transition_matrix = pickle.load(f)

    transition_probs_str = ""
    actions_str = ""

    actions_list = ', '.join(f'"{i}"' for i in range(n_actions))

    i = 1

    for t in range(n_timesteps):
        for s in range(n_states):
            probs = transition_matrix[(t, s)]
            transition_probs_str += f"{create_transition_prob_string_compressed(probs[0], probs[1])}\n"
            actions_str += f"""{actions_list} => prob{i}, """
            i += 1

    all_rewards = np.repeat(rewards, n_timesteps)

    string_array = [str(x) for x in all_rewards]
    rewards_str = "[" + ", ".join(string_array) + "]"

    transition_probs_code = f"""
    using IntervalMDP
    using JLD2
    using SparseArrays

    {transition_probs_str}

    transition_probs = [{actions_str}]

    initial_states = [Int32({initial_state + 1})]

    mdp = IntervalMarkovDecisionProcess(transition_probs, initial_states)
    """

    interval_mdp_code = f"""
    discount_factor = 1.0
    V_mins = []
    V_maxs = []

    for i in 1:{n_timesteps - 1}
        prop = FiniteTimeReward({rewards_str}, discount_factor, i)

        spec = Specification(prop, Pessimistic, Maximize)
        problem = Problem(mdp, spec)
        V_min, k, residual = value_iteration(problem)
        V_min = Array(V_min)
        push!(V_mins, V_min)

        spec = Specification(prop, Optimistic, Maximize)
        problem = Problem(mdp, spec)
        V_max, k, residual = value_iteration(problem)
        V_max = Array(V_max)
        push!(V_maxs, V_max)    
    end

    JLD2.save("ICFMDPs/{mdp_name}_value_pessimistic.jld2", "data", V_mins)
    JLD2.save("ICFMDPs/{mdp_name}_value_optimistic.jld2", "data", V_maxs)

    time_horizon = {n_timesteps - 1}

    prop = FiniteTimeReward({rewards_str}, discount_factor, time_horizon)

    spec = Specification(prop, Pessimistic, Maximize)
    problem = Problem(mdp, spec)
    V, k, residual = value_iteration(problem)
    policy = control_synthesis(problem)

    println(policy)

    JLD2.save("ICFMDPs/{mdp_name}.jld2", "data", policy)
    """

    generate_julia_file(filename, interval_mdp_code, transition_probs_code)
    print(f"Julia file '{filename}' generated successfully.")
