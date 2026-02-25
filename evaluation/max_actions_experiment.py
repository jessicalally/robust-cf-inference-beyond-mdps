from collections import defaultdict
import environments
from itertools import product
from evaluation.path_sampling import *
from evaluation.value_functions import *
from evaluation.compare_bounds import *
from src import simulate_counts, learn_imdp_pac_guaranteed, format_transition_matrix_for_julia, convert_transition_matrix_to_julia_imdp
from src import ParallelMultiStepCFBoundCalculatorApprox as IMDPMultiStepCFBoundCalculatorApprox
from src import ParallelMultiStepCFBoundCalculatorTight as IMDPMultiStepCFBoundCalculatorTight
from src import ParallelMultiStepCFBoundCalculator as MDPMultiStepCFBoundCalculator
from src import GumbelMaxCFBoundCalculatorIMDP
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import time

def plot_outcome_matrix(mdp, outcome_matrix, filename):
    outcome_matrix /= outcome_matrix.sum()
    
    plt.figure(figsize=(10, 10))
    plt.rcParams.update({'font.size': 25})
    ax = sns.heatmap(outcome_matrix, annot=True, 
                    annot_kws={'fontsize': 25},
                    fmt='.0%', cbar=False, cmap='Blues')
    ax.set_xlabel("\nCounterfactual Outcome")
    ax.set_ylabel("Observed Outcome\n")
    ax.set_xticklabels(['Failure', 'Suboptimal.', 'Success.'])
    ax.set_yticklabels(['Failure', 'Suboptimal.', 'Success.'])
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_verticalalignment('center')
   
    plt.savefig(f"max_actions_results/{mdp.env_name}-{filename}-swaps.pdf", bbox_inches='tight')
    plt.close()


def generate_data_max_actions_experiment(mdp, max_t, init_state, possible_trajectories=[100000], possible_gammas=[0.1]):
    run_gumbel_experiments = not(isinstance(mdp, environments.SepsisMDP) or isinstance(mdp, environments.AircraftMDP))
    mdp.init_state = init_state

    # Generate IMDP from n_runs.
    state_action_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    prev_runs = 0
    simulated_imdps = []

    for n_runs in possible_trajectories:
        # Generate IMDP from simulations.
        state_action_counts, transition_counts, _ = simulate_counts(mdp, prev_state_action_counts=state_action_counts, prev_transition_counts=transition_counts, num_episodes=n_runs-prev_runs, max_steps=max_t, seed=42, init_state=mdp.init_state)
        prev_runs = n_runs

        for gamma in possible_gammas:
            simulated_imdp = learn_imdp_pac_guaranteed(mdp, state_action_counts, transition_counts, num_episodes=n_runs-prev_runs, max_steps=max_t, gamma=gamma, method="clopper-pearson")
            simulated_imdps.append(simulated_imdp)

    hyperparameters = list(product(possible_trajectories, possible_gammas))

    simulated_cf_imdps_approx = []
    simulated_cf_imdps_tight = []
    true_cf_imdps = []

    # Sample paths in the MDP under a fixed, suboptimal behaviour policy.
    behaviour_policy = mdp.policy_with_randomization(mdp.optimal_policy, 0.5)
    sampled_paths = sample_paths(mdp, behaviour_policy, max_t, n_paths=100)

    elapsed_times_approx = {}
    elapsed_times_tight = {}
    elapsed_times_gumbel = {}
    elapsed_times_exact = []

    for (max_transitions, gamma) in hyperparameters:
        elapsed_times_approx[f"{(max_transitions, gamma)}"] = []
        elapsed_times_tight[f"{(max_transitions, gamma)}"] = []
        elapsed_times_gumbel[f"{(max_transitions, gamma)}"] = []

    for i, sampled_path in enumerate(sampled_paths):        
        print(f"i={i}, sampled_path={sampled_path}")
        
        simulated_cf_imdps_approx_this_path = []
        simulated_cf_imdps_tight_this_path = []

        for (max_transitions, gamma) in hyperparameters:
            # Generate approx CFIMDP from simulated IMDP.
            start = time.time()
            cf_bound_calculator = IMDPMultiStepCFBoundCalculatorApprox(mdp, simulated_imdp)
            simulated_cf_imdp_approx = cf_bound_calculator.calculate_bounds(sampled_path, n_jobs=32)
            end = time.time()
            elapsed_time_approx = end - start
            elapsed_times_approx[f"{(max_transitions, gamma)}"].append(elapsed_time_approx)

            # Save approx CFIMDP to Julia file.
            simulated_julia_cf_imdp = format_transition_matrix_for_julia(simulated_cf_imdp_approx, max_t, mdp.n_states, mdp.n_actions)

            convert_transition_matrix_to_julia_imdp(mdp.env_name, mdp, simulated_julia_cf_imdp, max_t, filename=f"MDPs/{mdp.env_name}/pac/simulated_{mdp.env_name}_{max_transitions}_{gamma}_max_actions_{i}_approx.jl", policy_header=f"{mdp.env_name}_max_actions_{max_transitions}_{gamma}_{i}_approx", observed_path=sampled_path)

            simulated_cf_imdps_approx_this_path.append(simulated_cf_imdp_approx)

            # Generate tight CFIMDP from simulated IMDP.
            start = time.time()
            cf_bound_calculator = IMDPMultiStepCFBoundCalculatorTight(mdp, simulated_imdp)
            simulated_cf_imdp_tight = cf_bound_calculator.calculate_bounds(sampled_path, n_jobs=32)
            end = time.time()
            elapsed_time_tight = end - start
            elapsed_times_tight[f"{(max_transitions, gamma)}"].append(elapsed_time_tight)

            average_width_diff_this_mdp, average_lb_diff_this_mdp, average_non_zero_diff_this_mdp, total_percentage_affected_this_mdp, approx_non_zero_affected_this_mdp = compare_approx_vs_tight_bounds_mdp(simulated_cf_imdp_approx, simulated_cf_imdp_tight)

            with open(f"results/{mdp.env_name}/pac/{mdp.env_name}_approx_vs_tight_max_actions.txt", "a") as f:
                f.write(f"Path={i} ##\n\n")
                f.write(f"Percentage affected = {total_percentage_affected_this_mdp * 100}%\n")
                f.write(f"Percentage nonzero transitions affected = {approx_non_zero_affected_this_mdp * 100}%\n")
                f.write(f"Average width diff = {average_width_diff_this_mdp}\n")
                f.write(f"Average LB diff = {average_lb_diff_this_mdp}\n")
                f.write(f"Average non-zero LB diff = {average_non_zero_diff_this_mdp}\n")
            
            simulated_julia_cf_imdp = format_transition_matrix_for_julia(simulated_cf_imdp_tight, max_t, mdp.n_states, mdp.n_actions)

            # Save tight CFIMDP to Julia file.
            convert_transition_matrix_to_julia_imdp(mdp.env_name, mdp, simulated_julia_cf_imdp, max_t, filename=f"MDPs/{mdp.env_name}/pac/simulated_{mdp.env_name}_{max_transitions}_{gamma}_max_actions_{i}_tight.jl", policy_header=f"{mdp.env_name}_max_actions_{max_transitions}_{gamma}_{i}_tight", observed_path=sampled_path)

            simulated_cf_imdps_tight_this_path.append(simulated_cf_imdp_tight)

            # Generate Gumbel-max CFIMDP from simulated IMDP.
            if run_gumbel_experiments:
                start = time.time()
                cf_bound_calculator = GumbelMaxCFBoundCalculatorIMDP(mdp, simulated_imdp, n_timesteps=max_t, n_states=mdp.n_states, n_actions=mdp.n_actions)
                simulated_cf_imdp_gumbel = cf_bound_calculator.run_gumbel_sampling(sampled_path)
                end = time.time()
                elapsed_time = end - start
                elapsed_times_gumbel[f"{(max_transitions, gamma)}"].append(elapsed_time)

                simulated_julia_cf_imdp = format_transition_matrix_for_julia(simulated_cf_imdp_gumbel, max_t, mdp.n_states, mdp.n_actions)

                # Save Gumbel-max CFIMDP to Julia file.
                convert_transition_matrix_to_julia_imdp(mdp.env_name, mdp, simulated_julia_cf_imdp, max_t, filename=f"MDPs/{mdp.env_name}/pac/simulated_{mdp.env_name}_{max_transitions}_{gamma}_max_actions_{i}_gumbel.jl", policy_header=f"{mdp.env_name}_max_actions_{max_transitions}_{gamma}_{i}_gumbel", observed_path=sampled_path)

        simulated_cf_imdps_approx.append(simulated_cf_imdps_approx_this_path)
        simulated_cf_imdps_tight.append(simulated_cf_imdps_tight_this_path)

        start = time.time()
        cf_bound_calculator = MDPMultiStepCFBoundCalculator(mdp, mdp.transition_matrix)
        true_cf_imdp = cf_bound_calculator.calculate_bounds(sampled_path)
        end = time.time()
        elapsed_time_exact = end - start
        elapsed_times_exact.append(elapsed_time_exact)

        with open(f"results/{mdp.env_name}/pac/compare_tight_vs_approx_execution_time_max_actions.txt", "a") as f:
            f.write(f"Exact = {elapsed_time_exact}\n")

        true_julia_cf_imdp = format_transition_matrix_for_julia(true_cf_imdp, max_t, mdp.n_states, mdp.n_actions)

        convert_transition_matrix_to_julia_imdp(mdp.env_name, mdp, true_julia_cf_imdp, max_t, filename=f"MDPs/{mdp.env_name}/pac/simulated_{mdp.env_name}_true_max_actions_{i}.jl", policy_header=f"{mdp.env_name}_true_max_actions_{i}", observed_path=sampled_path)
    
        true_cf_imdps.append(true_cf_imdp)

    with open(f"results/{mdp.env_name}/pac/compare_tight_vs_approx_execution_time_max_actions.txt", "a") as f:
        f.write(f"## IMDP -> CFIMDP EXECUTION TIME ##\n")

        for (max_transitions, gamma) in hyperparameters:
            f.write(f"Approx = {np.mean(np.array(elapsed_times_approx[f'{(max_transitions, gamma)}']))}+-{np.std(np.array(elapsed_times_approx[f'{(max_transitions, gamma)}']))}\n")
            f.write(f"Tight = {np.mean(np.array(elapsed_times_tight[f'{(max_transitions, gamma)}']))}+-{np.std(np.array(elapsed_times_tight[f'{(max_transitions, gamma)}']))}\n")
            f.write(f"Gumbel = {np.mean(np.array(elapsed_times_gumbel[f'{(max_transitions, gamma)}']))}+-{np.std(np.array(elapsed_times_gumbel[f'{(max_transitions, gamma)}']))}\n")

        f.write(f"Exact = {np.mean(np.array(elapsed_times_exact))}+-{np.std(np.array(elapsed_times_exact))}\n")

    with open(f"{mdp.env_name}_max_actions_observed_paths.pickle", "wb") as f:
        pickle.dump((sampled_paths, behaviour_policy, np.array(simulated_cf_imdps_approx), np.array(true_cf_imdps), np.array(simulated_cf_imdps_tight)), f)


def load_generated_policies(filename, n_timesteps, n_states, actions):
    pi = np.zeros(shape=(n_timesteps, n_timesteps, n_states))

    with h5py.File(filename, "r") as file:
        for max_actions in range(n_timesteps):
            ref = file["data"][()][max_actions]
            target = file[ref]
            arr = np.array(target).astype(int)

            for t in range(n_timesteps):
                for s in range(n_states):
                    pi[max_actions, t, s] = arr[n_timesteps-(t+1), (t*n_states)+s]-1
                    
                    # Loading check.
                    if max_actions == 0:
                        assert(pi[max_actions, t, s] == actions[t])

    return pi.astype(int)


def map_value_to_outcome(mdp, value):
    if isinstance(mdp, environments.SepsisMDP):
        if value < -1000:
            # Failure
            return 0
        elif value >= 0 and value < 1000:
            # Suboptimal
            return 1
        
        # Success
        assert(value >= 1000)
        return 2
    
    if value < 0:
        # On average, paths reach the catastrophic, terminal state.
        return 0
    elif 0 <= value < 100:
        # Paths are suboptimal, don't reach goal state.
        return 1
    
    assert(value >= 100)
    return 2


def run_max_actions_experiment(mdp, max_t, init_state, possible_trajectories=[100000], possible_gammas=[0.1]):
    with open(f"{mdp.env_name}_max_actions_observed_paths.pickle", "rb") as f:
        sampled_paths, behaviour_policy, simulated_cf_imdps_approx, true_cf_imdps, simulated_cf_imdps_tight = pickle.load(f)

    run_gumbel_experiments = not(isinstance(mdp, environments.SepsisMDP) or isinstance(mdp, environments.AircraftMDP))
    mdp.init_state = init_state

    hyperparameters = list(product(possible_trajectories, possible_gammas))

    # Evaluate width diffs.
    compare_approx_vs_tight_bounds(mdp.env_name, hyperparameters, sampled_paths, simulated_cf_imdps_approx, simulated_cf_imdps_tight)

    # Evaluate value functions.
    outcome_matrices_values = {}
    values = {}
    tight_values = {}
    gumbel_robustness_values = {}
    approx_robustness_values = {}

    for i, sampled_path in enumerate(sampled_paths):
        print(f"sampled={sampled_path}, i={i}")

        # Compute observed outcome.
        def get_outcome_idx(total_reward):
            if isinstance(mdp, environments.SepsisMDP):
                if total_reward < -1000:
                    # Failure
                    return 0
                elif total_reward >= 0 and total_reward < 1000:
                    # Suboptimal
                    return 1
                else:
                    # Success
                    assert(total_reward >= 1000)
                    return 2

            else:
                if total_reward < 0:
                    # Failure
                    return 0
                elif total_reward >= 0 and total_reward < 100:
                    # Suboptimal
                    return 1
                else:
                    # Success
                    assert(total_reward >= 100)
                    return 2

        total_observed_reward = np.sum(sampled_path[:, 3])
        observed_outcome = get_outcome_idx(total_observed_reward)
        actions = sampled_path[:, 2]

        for j, (n_runs, gamma) in enumerate(hyperparameters):
            ## MAX ACTIONS EXPERIMENT

            # Get policies.
            policy_file = f"ICFMDPs/{mdp.env_name}/{mdp.env_name}_max_actions_{n_runs}_{gamma}_{i}_approx_policies.jld2"
            policies = load_generated_policies(policy_file, max_t, mdp.n_states, actions)

            # Evaluate the estimated value of the generated CF policy on the approx simulated CFIMDP.
            value_functions = policy_eval_imdp_deterministic_max_actions(sampled_path, simulated_cf_imdps_approx[i, j], mdp, policies, max_t, gamma=1.0)
            values_initial_state = value_functions[:, 0, mdp.init_state]
            print(f"initial state values = {values_initial_state}")

            # Maximum number of actions that can be changed from observation.
            for k in range(max_t):
                cf_outcome = map_value_to_outcome(mdp, values_initial_state[k])

                assert(cf_outcome >= observed_outcome)

                if not f"({n_runs}, {gamma}, {k})" in outcome_matrices_values:
                    outcome_matrices_values[f"({n_runs}, {gamma}, {k})"] = np.zeros(shape=(3, 3))
                    values[f"({n_runs}, {gamma}, {k})"] = []
                    tight_values[f"({n_runs}, {gamma}, {k})"] = []
                    approx_robustness_values[f"({n_runs}, {gamma}, {k})"] = []
                    gumbel_robustness_values[f"({n_runs}, {gamma}, {k})"] = []

                outcome_matrices_values[f"({n_runs}, {gamma}, {k})"][observed_outcome, cf_outcome] += 1
                values[f"({n_runs}, {gamma}, {k})"].append(values_initial_state[k])

            policy_file = f"ICFMDPs/{mdp.env_name}/{mdp.env_name}_max_actions_{n_runs}_{gamma}_{i}_tight_policies.jld2"
            policies = load_generated_policies(policy_file, max_t, mdp.n_states, actions)

            # Evaluate the estimated value of the generated CF policy on the tight simulated CFIMDP.
            value_functions = policy_eval_imdp_deterministic_max_actions(sampled_path, simulated_cf_imdps_tight[i, j], mdp, policies, max_t, gamma=1.0)
            values_initial_state = value_functions[:, 0, mdp.init_state]

            for k in range(max_t):
                tight_values[f"({n_runs}, {gamma}, {k})"].append(values_initial_state[k])

            ## ROBUSTNESS EXPERIMENT

            # Evaluate robustness of approx CFIMDP policy on true CF IMDP.
            policy_file = f"ICFMDPs/{mdp.env_name}/{mdp.env_name}_max_actions_{n_runs}_{gamma}_{i}_approx_policies.jld2"
            policies = load_generated_policies(policy_file, max_t, mdp.n_states, actions)

            value_functions = policy_eval_imdp_deterministic_max_actions(sampled_path, true_cf_imdps[i], mdp, policies, max_t, gamma=1.0)
            values_initial_state = value_functions[:, 0, mdp.init_state]

            for k in range(max_t):
                approx_robustness_values[f"({n_runs}, {gamma}, {k})"].append(values_initial_state[k])

            # Evaluate robustness of Gumbel-max on true CF IMDP.
            if run_gumbel_experiments:
                policy_file = f"ICFMDPs/{mdp.env_name}/{mdp.env_name}_max_actions_{n_runs}_{gamma}_{i}_gumbel_policies.jld2"
                policies = load_generated_policies(policy_file, max_t, mdp.n_states, actions)

                value_functions = policy_eval_imdp_deterministic_max_actions(sampled_path, true_cf_imdps[i], mdp, policies, max_t, gamma=1.0)
                values_initial_state = value_functions[:, 0, mdp.init_state]

                for k in range(max_t):
                    gumbel_robustness_values[f"({n_runs}, {gamma}, {k})"].append(values_initial_state[k])

        # Compute outcome swaps.
        policy_file = f"ICFMDPs/{mdp.env_name}/{mdp.env_name}_true_max_actions_{i}_policies.jld2"
        policies = load_generated_policies(policy_file, max_t, mdp.n_states, actions)

        value_functions = policy_eval_imdp_deterministic_max_actions(sampled_path, true_cf_imdps[i], mdp, policies, max_t, gamma=1.0)
        values_initial_state = value_functions[:, 0, mdp.init_state]

        # Maximum number of actions that can be changed from observation.
        for k in range(max_t):
            cf_outcome = map_value_to_outcome(mdp, values_initial_state[k])

            if not f"exact, {k}" in outcome_matrices_values:
                outcome_matrices_values[f"exact, {k}"] = np.zeros(shape=(3, 3))
                values[f"exact, {k}"] = []

            outcome_matrices_values[f"exact, {k}"][observed_outcome, cf_outcome] += 1
            values[f"exact, {k}"].append(values_initial_state[k])


    for k in range(max_t):
        for (n_runs, gamma) in hyperparameters:
            plot_outcome_matrix(mdp, outcome_matrices_values[f"({n_runs}, {gamma}, {k})"], f"({n_runs}, {gamma}, {k})")
            
            values_these_hyperparameters = np.array(values[f"({n_runs}, {gamma}, {k})"])
            mean_value = np.mean(values_these_hyperparameters)
            std_value = np.std(values_these_hyperparameters)

            tight_values_these_hyperparameters = np.array(tight_values[f"({n_runs}, {gamma}, {k})"])
            tight_mean_value = np.mean(tight_values_these_hyperparameters)
            tight_std_value = np.std(tight_values_these_hyperparameters)

            approx_values_robustness_these_hyperparameters = np.array(approx_robustness_values[f"({n_runs}, {gamma}, {k})"])
            approx_robustness_mean_value = np.mean(approx_values_robustness_these_hyperparameters)
            approx_robustness_std_value = np.std(approx_values_robustness_these_hyperparameters)

            gumbel_values_robustness_these_hyperparameters = np.array(gumbel_robustness_values[f"({n_runs}, {gamma}, {k})"])
            gumbel_robustness_mean_value = np.mean(gumbel_values_robustness_these_hyperparameters)
            gumbel_robustness_std_value = np.std(gumbel_values_robustness_these_hyperparameters)

            with open(f"results/{mdp.env_name}/pac/robustness.txt", "a") as f:
                f.write(f"HYPERPARAMETERS = {(n_runs, gamma)}\n")
                f.write(f"approx max actions changed = {k}, value={mean_value}+-{std_value}\n")
                f.write(f"tight max actions changed = {k}, value={tight_mean_value}+-{tight_std_value}\n")
                f.write(f"approx robustness max actions changed = {k}, value={approx_robustness_mean_value}+-{approx_robustness_std_value}\n")
                f.write(f"gumbel robustness max actions changed = {k}, value={gumbel_robustness_mean_value}+-{gumbel_robustness_std_value}\n")

        values_exact = np.array(values[f"exact, {k}"])
        mean_value = np.mean(values_exact)
        std_value = np.std(values_exact)

        with open(f"results/{mdp.env_name}/pac/robustness.txt", "a") as f:
            f.write(f"HYPERPARAMETERS = exact\n")
            f.write(f"max actions changed = {k}, value={mean_value}+-{std_value}\n")

        plot_outcome_matrix(mdp, outcome_matrices_values[f"exact, {k}"], f"exact, {k}")
