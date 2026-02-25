import h5py
from joblib import Parallel, delayed
import numpy as np

TRANSITION_PROB_COUNTER = 1
ACTION_MAP = {}

def _format_single_ts(t, s, interval_CF_MDP, n_timesteps, n_states, n_actions):
    lower_transition_probs = np.zeros((n_states * (n_timesteps + 1), n_actions), dtype=np.float64)
    upper_transition_probs = np.zeros((n_states * (n_timesteps + 1), n_actions), dtype=np.float64)

    for a in range(n_actions):
        for s_prime in range(n_states):
            bounds = interval_CF_MDP[t, s, a, s_prime]
            lower_transition_probs[((t + 1) * n_states) + s_prime, a] = bounds[0]
            upper_transition_probs[((t + 1) * n_states) + s_prime, a] = bounds[1]

    return ((t, s), (lower_transition_probs, upper_transition_probs))


def _format_sink_state(ts_sink, n_timesteps, n_states, n_actions):
    lower_transition_probs = np.zeros((n_states * (n_timesteps + 1), n_actions), dtype=np.float64)
    upper_transition_probs = np.zeros((n_states * (n_timesteps + 1), n_actions), dtype=np.float64)

    lower_transition_probs[(n_timesteps * n_states) + ts_sink, :] = 1.0
    upper_transition_probs[(n_timesteps * n_states) + ts_sink, :] = 1.0

    return ((n_timesteps, ts_sink), (lower_transition_probs, upper_transition_probs))



def format_transition_matrix_for_julia(interval_CF_MDP, n_timesteps, n_states, n_actions, n_jobs=8):
    # Parallelise the inner loop over (t, s)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_format_single_ts)(t, s, interval_CF_MDP, n_timesteps, n_states, n_actions)
        for t in range(n_timesteps)
        for s in range(n_states)
    )

    # Add the final sink states (one per s at t = n_timesteps)
    sink_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_format_sink_state)(s, n_timesteps, n_states, n_actions)
        for s in range(n_states)
    )

    # Combine all results into a single dict
    transition_matrix = dict(results + sink_results)

    return transition_matrix


def load_generated_policy(filename, n_timesteps, n_states):
    pi = np.zeros(shape=(n_timesteps, n_states))

    with h5py.File(filename, "r") as file:
        ref = file["data"][()][0]
        target = file[ref]
        arr = target[:]

        for t in range(n_timesteps):
            ref = arr[t]
            res = np.array(file[ref])
            int_array = [int(byte_str) for byte_str in res]
            
            for s in range(n_states):
                pi[t, s] = int_array[(t*n_states)+s] - 1
    
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


def load_value_function_max_actions(filename, n_timesteps, n_states):
    V = np.zeros(shape=(n_timesteps, n_timesteps, n_states))

    with h5py.File(filename, "r") as file:
        data = file["data"]

        for t in range(1, n_timesteps+1):
            ref = data[n_timesteps-t]
            res = np.array(file[ref])
            max_poss_actions = res.shape[0]
      
            # float_array = [float(byte_str) for byte_str in res]
            float_array = np.array([[float(byte_str) for byte_str in row] for row in res])

            for max_actions_change in range(max_poss_actions):
                for s in range(n_states):
                    V[t-1, max_actions_change-1, s] = float_array[max_actions_change, ((t-1) * n_states)+s]
        
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

    lower_str = "\n        ".join([
        " ".join(f"{x:.5g}" for x in row) for row in lower_probs
    ])

    upper_str = "\n        ".join([
        " ".join(f"{x:.5g}" for x in row) for row in upper_probs
    ])

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

def format_float(x):
    s = f"{x:.16f}"
    return s if "." in s or "e" in s else s + ".0"

def create_transition_prob_string_compressed(t, s, n_states, lower_probs, upper_probs, valid_actions):
    global TRANSITION_PROB_COUNTER
    global ACTION_MAP

    lower_str_parts = []
    upper_str_parts = []

    state_idx = (t * n_states) + s
    action_idx = 0

    for action in range(lower_probs.shape[1]):
        if valid_actions[s, action]:
            ACTION_MAP[(state_idx+1, action_idx+1)] = action+1 # Julia is 1-indexed.

            non_zero_probs_indices = np.nonzero(upper_probs[:, action])[0] # use upper bound indices because some states might have LB=0
            # Probs are correct to 16dp
            upper_probs = np.round(upper_probs, decimals=16)
            lower_probs = np.round(lower_probs, decimals=16)

            non_zero_probs_lower = ", ".join(format_float(x) for x in lower_probs[non_zero_probs_indices, action])      
            non_zero_probs_upper = ", ".join(format_float(x) for x in upper_probs[non_zero_probs_indices, action])

            non_zero_probs_indices = [idx+1 for idx in non_zero_probs_indices]
            non_zero_probs_indices = ", ".join(map(str, non_zero_probs_indices))
            
            lower_str_parts.append(f"\n\t\t\tSparseVector({lower_probs.shape[0]}, [{non_zero_probs_indices}], [{non_zero_probs_lower}]),")
            upper_str_parts.append(f"\n\t\t\tSparseVector({upper_probs.shape[0]}, [{non_zero_probs_indices}], [{non_zero_probs_upper}]),")

            action_idx += 1

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


def convert_transition_matrix_to_julia_imdp(env_name, env, transition_matrix, n_timesteps, filename, policy_header, observed_path=None):
    global TRANSITION_PROB_COUNTER
    TRANSITION_PROB_COUNTER = 1

    global ACTION_MAP
    ACTION_MAP = {}

    transition_probs_str = ""
    actions_str = ""

    i = 1

    for t in range(n_timesteps+1):
        for s in range(env.n_states):
            probs = transition_matrix[(t,s)]
            transition_probs_str += f"{create_transition_prob_string_compressed(t, s, env.n_states, probs[0], probs[1], env.valid_actions)}\n"
            actions_str += f"prob{i}, "
            i += 1

    rewards = np.zeros((n_timesteps+1) * env.n_states)
    for s in range((n_timesteps+1) * env.n_states):
        rewards[s] = env.state_rewards[s % env.n_states]

    string_array = [str(x) for x in rewards]
    rewards_str = "[" + ", ".join(string_array) + "]"

    if observed_path is None:
        observed_actions_array = [str(a) for a in env.suboptimal_trajectory[:, 2] for _ in range(env.n_states)]
    else:
        observed_actions_array = [str(a) for a in observed_path[:, 2] for _ in range(env.n_states)]

    # We have to add dummy observed actions for the sink states, so we pick the default action "0"
    observed_actions_array += ["0" for _ in range(env.n_states)]
    observed_actions_str = "[" + ", ".join(observed_actions_array) + "]"

    action_map_str = "action_map = Dict(\n"

    for (s, col), a in ACTION_MAP.items():
        action_map_str += f"    ({s},{col}) => {a},\n"
    action_map_str += ")\n"

    transition_probs_code = f"""
    using IntervalMDP
    using JLD2
    using SparseArrays

    {transition_probs_str}

    transition_probs = [{actions_str}]

    initial_states = [Int32({env.init_state + 1})]

    mdp = IntervalMarkovDecisionProcess(transition_probs, initial_states)
    """

    interval_mdp_code = f"""
    discount_factor = 1.0
    V_mins = []
    V_maxs = []
    time_horizon = {n_timesteps}

    prop = FiniteTimeReward({rewards_str}, discount_factor, time_horizon)

    spec = Specification(prop, Pessimistic, Maximize)
    problem = ControlSynthesisProblem(mdp, spec)
    policy, V, k, residual = solve(problem)
    
    value_init_state = V[initial_states]
    println(value_init_state)

    JLD2.save("ICFMDPs/{env_name}/{policy_header}_policy.jld2", "data", policy)
    
    solutions = solve_max_changes(problem, time_horizon, {observed_actions_str})
    values = []
    policies = []

	for i in 1:time_horizon+1
		policy_i_actions, V_i, k_i, residual_i = solutions[i]
        value_init_state_i = V_i[initial_states][1]
    	push!(values, value_init_state_i)
        push!(policies, policy_i_actions)
	end

    JLD2.save("ICFMDPs/{env_name}/{policy_header}_policies.jld2", "data", policies)

    println(values)
    """

    generate_julia_file(filename, interval_mdp_code, transition_probs_code)
    print(f"Julia file '{filename}' generated successfully.")
