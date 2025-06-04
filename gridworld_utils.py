from decimal import Decimal, getcontext
import numpy as np
import pickle

TRANSITION_PROB_COUNTER = 1

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
    s = f"{x:.15f}"
    return s if "." in s or "e" in s else s + ".0"

def create_transition_prob_string_compressed(lower_probs, upper_probs):
    global TRANSITION_PROB_COUNTER

    lower_str_parts = []
    upper_str_parts = []

    for col in range(lower_probs.shape[1]):
        non_zero_probs_indices = np.nonzero(upper_probs[:, col])[0] # use upper bound indicies because some states might have LB=0
        
        # Probs are correct to 15dp
        upper_probs = np.round(upper_probs, decimals=15)
        lower_probs = np.round(lower_probs, decimals=15)

        total = np.sum(lower_probs[non_zero_probs_indices, col])

        if total > 1.0:
            lower_probs[non_zero_probs_indices[0], col] -= 1e-15    

        total = np.sum(upper_probs[non_zero_probs_indices, col])

        if total < 1.0:
            upper_probs[non_zero_probs_indices[0], col] += 1e-15    

        non_zero_probs_lower = ", ".join(format_float(x) for x in lower_probs[non_zero_probs_indices, col])      
        non_zero_probs_upper = ", ".join(format_float(x) for x in upper_probs[non_zero_probs_indices, col])

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


def convert_transition_matrix_to_julia_imdp(transition_matrix, num_transitions, delta, tra_filename = f"transition_matrices/gridworld_tra.pickle", filename = f"MDPs/gridworld.jl"):
    global TRANSITION_PROB_COUNTER
    TRANSITION_PROB_COUNTER = 1

    # with open(tra_filename, "rb") as f:
    #     transition_matrix = pickle.load(f)

    transition_probs_str = ""
    actions_str = ""

    i = 1

    for t in range(11):
        for s in range(16):
            probs = transition_matrix[(t,s)]
            transition_probs_str += f"{create_transition_prob_string_compressed(probs[0], probs[1])}\n"
            actions_str += f"""["0", "1", "2", "3"] => prob{i}, """
            i += 1

    rewards = np.zeros(11 * 16)
    for s in range(11 * 16):
        if s%16 == 6:
            rewards[s] = -100
        elif s%16 == 15:
            rewards[s] = 100
        elif s%16 in [1, 4, 5]:
            rewards[s] = 1.0
        elif s%16 in [2, 8, 9]:
            rewards[s] = 2.0
        elif s%16 in [3, 10, 12]:
            rewards[s] = 3.0
        elif s%16 in [7, 13]:
            rewards[s]= 4.0
        elif s%16 in [11, 14]:
            rewards[s] = 5.0

    string_array = [str(x) for x in rewards]
    rewards_str = "[" + ", ".join(string_array) + "]"

    transition_probs_code = f"""
    using IntervalMDP
    using JLD2
    using SparseArrays

    {transition_probs_str}

    transition_probs = [{actions_str}]

    initial_states = [Int32(1)]

    mdp = IntervalMarkovDecisionProcess(transition_probs, initial_states)
    """

    interval_mdp_code = f"""
    discount_factor = 1.0
    V_mins = []
    V_maxs = []

    for i in 1:10
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

    JLD2.save("ICFMDPs/gridworld_value_pessimistic_{num_transitions}_{delta}.jld2", "data", V_mins)
    JLD2.save("ICFMDPs/gridworld_value_optimistic_{num_transitions}_{delta}.jld2", "data", V_maxs)

    time_horizon = 10

    prop = FiniteTimeReward({rewards_str}, discount_factor, time_horizon)

    spec = Specification(prop, Pessimistic, Maximize)
    problem = Problem(mdp, spec)
    V, k, residual = value_iteration(problem)
    policy = control_synthesis(problem)

    println(policy)

    JLD2.save("ICFMDPs/gridworld_policy_{num_transitions}_{delta}.jld2", "data", policy)
    """

    generate_julia_file(filename, interval_mdp_code, transition_probs_code)
    print(f"Julia file '{filename}' generated successfully.")
