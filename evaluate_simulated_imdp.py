from generate_imdp import learn_imdp, normalise_imdp
from gridworld import MDP
from gridworld_utils import convert_transition_matrix_to_julia_imdp
import gurobipy as gp
from imdp_cf_bounds import MultiStepCFBoundCalculatorTight as IMDPMultiStepCFBoundCalculatorTight
from imdp_cf_bounds import MultiStepCFBoundCalculatorApprox as IMDPMultiStepCFBoundCalculatorApprox
import itertools
import matplotlib.pyplot as plt
from mdp_cf_bounds import MultiStepCFBoundCalculator as MDPMultiStepCFBoundCalculator
import numpy as np
import pickle
import sys
import time
from utils import save_imdp_to_file, format_transition_matrix_for_julia, load_generated_policy, load_value_function


def sample_probabilities_within_intervals(lower, upper):    
    while True:
        # Sample uniformly within the given bounds.
        sampled_probs = np.random.uniform(lower, upper)
        
        # Normalise the sampled probabilities to sum to 1.
        sampled_probs /= np.sum(sampled_probs)
        sampled_probs /= np.sum(sampled_probs)

        # Allowed error due to floating-point errors.
        epsilon = 1e-13

        # Check if the normalized probabilities still lie within the bounds.
        if np.all(sampled_probs >= lower-epsilon) and np.all(sampled_probs <= upper+epsilon):
            return sampled_probs


def sample_CFMDP(interval_CF_MDP, n_timesteps=10, n_states=16, n_actions=4):
    CFMDP = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states))

    for t in range(n_timesteps):
        for s in range(n_states):
            for a in range(n_actions):
                CFMDP[t, s, a] = sample_probabilities_within_intervals(interval_CF_MDP[t, s, a, :, 0], interval_CF_MDP[t, s, a, :, 1])

    return CFMDP


def evaluate_policies(MDP_rewards, true_pi, simulated_pi, interval_CF_MDP, n_steps = 10):
    n_state = 4
    N_CFMDPS = 200
    N_TRAJECTORIES = 10000

    all_rewards_simulated_icfmdp = []
    all_rewards_icfmdp = []
    
    for k in range(N_CFMDPS):
        print(f"{k}/{N_CFMDPS}")
        CFMDP = sample_CFMDP(interval_CF_MDP)

        # Test with true ICFMDP policy.
        for j in range(N_TRAJECTORIES):
            print(f"{j}/{N_TRAJECTORIES}")
            trajectory = np.zeros((n_steps, n_state))
            current_state = 0

            for time_idx in range(n_steps):
                action = true_pi[time_idx, current_state]
                next_state = np.random.choice(16, size=1, p=CFMDP[time_idx, current_state, action])[0] 
                
                reward = MDP_rewards[current_state, action]
                trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
                current_state = next_state

            rewards = trajectory[:, 3]

            all_rewards_icfmdp.append(rewards)

        # Test with policy learned over simulated IMDP.
        for j in range(N_TRAJECTORIES):
            print(f"{j}/{N_TRAJECTORIES}")

            trajectory = np.zeros((n_steps, n_state))
            current_state = 0

            for time_idx in range(n_steps):
                action = simulated_pi[time_idx, current_state]
                next_state = np.random.choice(16, size=1, p=CFMDP[time_idx, current_state, action])[0] 
                reward = MDP_rewards[current_state, action]
                trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
                current_state = next_state

            rewards = trajectory[:, 3]
            all_rewards_simulated_icfmdp.append(rewards)

    return all_rewards_icfmdp, all_rewards_simulated_icfmdp


def main():
    if len(sys.argv) < 2:
        print("Usage: python gridworld.py <function_name>")
        sys.exit(1)
    
    function_name = sys.argv[1]

    mdp = MDP()

    possible_deltas = [0.01, 0.05, 0.1, 0.2]
    possible_transitions = [1000, 2500, 5000, 10000]
    max_steps = 20

    hyperparameters = list(itertools.product(possible_transitions, possible_deltas))
    
    if function_name == "train":
        for (max_transitions, delta) in hyperparameters:
            # Generate IMDP from simulations.
            simulated_imdp = learn_imdp(mdp, num_episodes=int(max_transitions/max_steps), max_steps=max_steps, delta=delta)

            # Generate observed path, and normalise IMDP.
            observed_path = mdp.sample_suboptimal_trajectory()
            simulated_imdp = normalise_imdp(simulated_imdp, observed_path)
            # save_imdp_to_file("gridworld_imdp.txt", simulated_imdp)

            # Generate CFIMDP from simulated IMDP.
            cf_bound_calculator = IMDPMultiStepCFBoundCalculatorApprox(simulated_imdp)
            simulated_cf_imdp = cf_bound_calculator.calculate_bounds(observed_path)

            with open(f"MDPs/simulated_cf_gridworld_{max_transitions}_{delta}.pickle", "wb") as f:
                pickle.dump(simulated_cf_imdp, f)
            
            with open(f"MDPs/simulated_gridworld_{max_transitions}_{delta}.pickle", "wb") as f:
                pickle.dump(simulated_imdp, f)

            simulated_julia_cf_imdp = format_transition_matrix_for_julia(simulated_cf_imdp, 10, 16, 4)
            convert_transition_matrix_to_julia_imdp(simulated_julia_cf_imdp, max_transitions, delta, filename=f"MDPs/simulated_gridworld_{max_transitions}_{delta}.jl")

        # Generate CFIMDP from true IMDP.
        cf_bound_calculator = MDPMultiStepCFBoundCalculator(mdp.transition_matrix)
        true_cf_imdp = cf_bound_calculator.calculate_bounds(observed_path)

        true_julia_cf_imdp = format_transition_matrix_for_julia(true_cf_imdp, 10, 16, 4)
        convert_transition_matrix_to_julia_imdp(true_julia_cf_imdp, "", "", filename=f"MDPs/true_gridworld.jl")


    # elif function_name == "generate_icfmdps":
    #     max_transitions = int(sys.argv[2])
    #     delta = float(sys.argv[3])

    #     # Generate IMDP from simulations.
    #     simulated_imdp = learn_imdp(mdp, num_episodes=int(max_transitions/max_steps), max_steps=max_steps, delta=delta)

    #     # Generate observed path, and normalise IMDP.
    #     observed_path = mdp.sample_suboptimal_trajectory()
    #     simulated_imdp = normalise_imdp(simulated_imdp, observed_path)

    #     # Generate approx CFIMDP from simulated IMDP.
    #     start = time.time()
    #     cf_bound_calculator = IMDPMultiStepCFBoundCalculatorApprox(simulated_imdp)
    #     simulated_cf_imdp = cf_bound_calculator.calculate_bounds(observed_path)
    #     end = time.time()
    #     elapsed_time = end - start
    #     print(elapsed_time)

    #     with open(f"MDPs/simulated_cf_gridworld_{max_transitions}_{delta}_approx.pickle", "wb") as f:
    #         pickle.dump(simulated_cf_imdp, f)

    #     # Generate tight CFIMDP from simulated IMDP.
    #     start = time.time()
    #     cf_bound_calculator = IMDPMultiStepCFBoundCalculatorTight(simulated_imdp)
    #     simulated_cf_imdp = cf_bound_calculator.calculate_bounds(observed_path)
    #     end = time.time()
    #     elapsed_time = end - start
    #     print(elapsed_time)

    #     with open(f"MDPs/simulated_cf_gridworld_{max_transitions}_{delta}_tight.pickle", "wb") as f:
    #         pickle.dump(simulated_cf_imdp, f)


    elif function_name == "compare_tight_vs_approx":
        # Compares the lower bounds produced by the tight optimisation procedure vs the approximate analytical solution.
        elapsed_times_approx = []
        elapsed_times_exact = []

        for (max_transitions, delta) in hyperparameters:
            with open(f"tight_vs_approx_execution_time.txt", "a") as f:
                f.write(f"## HYPERPARMETERS = {max_transitions}, {delta} ##\n\n")

            # Generate IMDP from simulations.
            simulated_imdp = learn_imdp(mdp, num_episodes=int(max_transitions/max_steps), max_steps=max_steps, delta=delta)

            # Generate observed path, and normalise IMDP.
            observed_path = mdp.sample_suboptimal_trajectory()
            simulated_imdp = normalise_imdp(simulated_imdp, observed_path)

            with open(f"MDPs/simulated_imdp.pickle", "wb") as f:
                pickle.dump(simulated_imdp, f)

            # Generate approx CFIMDP from simulated IMDP.
            start = time.time()
            cf_bound_calculator = IMDPMultiStepCFBoundCalculatorApprox(simulated_imdp)
            simulated_cf_imdp = cf_bound_calculator.calculate_bounds(observed_path)
            end = time.time()
            elapsed_time = end - start
            elapsed_times_approx.append(elapsed_time)

            with open(f"tight_vs_approx_execution_time.txt", "a") as f:
                f.write(f"## IMDP -> CFIMDP EXECUTION TIME ##\n")
                f.write(f"Approx MDP = {elapsed_time}\n")

            with open(f"MDPs/simulated_cf_gridworld_{max_transitions}_{delta}_approx.pickle", "wb") as f:
                pickle.dump(simulated_cf_imdp, f)

            # Generate tight CFIMDP from simulated IMDP.
            start = time.time()
            cf_bound_calculator = IMDPMultiStepCFBoundCalculatorTight(simulated_imdp)
            simulated_cf_imdp = cf_bound_calculator.calculate_bounds(observed_path)
            end = time.time()
            elapsed_time = end - start
            elapsed_times_exact.append(elapsed_time)

            with open(f"tight_vs_approx_execution_time.txt", "a") as f:
                f.write(f"## IMDP -> CFIMDP EXECUTION TIME ##\n")
                f.write(f"Tight MDP = {elapsed_time}\n")

            with open(f"MDPs/simulated_cf_gridworld_{max_transitions}_{delta}_tight.pickle", "wb") as f:
                pickle.dump(simulated_cf_imdp, f)

        with open(f"tight_vs_approx_execution_time.txt", "a") as f:
            f.write(f"## AVERAGE EXECUTION TIMES ##\n")
            f.write(f"Approx Approach = {np.mean(np.array(elapsed_times_approx))}\n")
            f.write(f"Exact Approach = {np.mean(np.array(elapsed_times_exact))}\n")       

        # Generate exact CFIMDP from true IMDP.
        start = time.time()
        cf_bound_calculator = MDPMultiStepCFBoundCalculator(mdp.transition_matrix)
        true_cf_imdp = cf_bound_calculator.calculate_bounds(observed_path)
        end = time.time()
        elapsed_time = end - start

        with open(f"MDPs/true_cf_gridworld.pickle", "wb") as f:
            pickle.dump(true_cf_imdp, f)

        with open(f"tight_vs_approx_execution_time.txt", "a") as f:
            f.write(f"## MDP -> CFIMDP EXECUTION TIME ##\n")
            f.write(f"Exact MDP = {elapsed_time}\n")


    elif function_name == "evaluate_diff_in_lower_bounds":
        for (max_transitions, delta) in hyperparameters:
            with open(f"MDPs/simulated_cf_gridworld_{max_transitions}_{delta}_approx.pickle", "rb") as f:
                approx_simulated_cf_imdp = pickle.load(f)

            with open(f"MDPs/simulated_cf_gridworld_{max_transitions}_{delta}_tight.pickle", "rb") as f:
                tight_simulated_cf_imdp = pickle.load(f)

            with open(f"MDPs/true_gridworld.pickle", "rb") as f:
                true_imdp = pickle.load(f)

            n_timesteps = approx_simulated_cf_imdp.shape[0]
            n_states = approx_simulated_cf_imdp.shape[1]
            n_actions = approx_simulated_cf_imdp.shape[2]

            width_diffs = []
            lb_diffs = []
            non_zero_lb_diffs = []

            for t in range(n_timesteps):
                for s in range(n_states):
                    for a in range(n_actions):
                        for s_prime in range(n_states):
                            # Assert UBs are the same.
                            assert(approx_simulated_cf_imdp[t, s, a, s_prime, 1] == tight_simulated_cf_imdp[t, s, a, s_prime, 1])

                            # Compare the bound widths.
                            approx_width = approx_simulated_cf_imdp[t, s, a, s_prime, 1] - approx_simulated_cf_imdp[t, s, a, s_prime, 0]
                            tight_width = tight_simulated_cf_imdp[t, s, a, s_prime, 1] - tight_simulated_cf_imdp[t, s, a, s_prime, 0]
                            width_diff = abs(approx_width - tight_width)
                            width_diffs.append(width_diff)

                            # Compare the lower bounds.
                            approx_lb = approx_simulated_cf_imdp[t, s, a, s_prime, 0]
                            tight_lb = tight_simulated_cf_imdp[t, s, a, s_prime, 0]

                            lb_diff = abs(tight_lb - approx_lb)
                            lb_diffs.append(lb_diff)

                            if not tight_lb == approx_lb:
                                print(f"tight={tight_lb}; approx={approx_lb}; true={true_imdp[s, a, s_prime]}")
                                non_zero_lb_diffs.append(lb_diff)

            average_width_diff = np.mean(np.array(width_diffs))
            average_lb_diff = np.mean(np.array(lb_diffs))
            average_non_zero_diff = np.mean(np.array(non_zero_lb_diffs))

            with open(f"gridworld_lb_diff.txt", "a") as f:
                f.write(f"## HYPERPARAMETERS = {max_transitions}, {delta} ##\n\n")
                f.write(f"Average width diff = {average_width_diff}\n")
                f.write(f"Average LB diff = {average_lb_diff}\n")
                f.write(f"Average non-zero LB diff = {average_non_zero_diff}\n\n")


    elif function_name == "test":
        all_rewards = []

        for i, (max_transitions, delta) in enumerate(hyperparameters):
            observed_path = mdp.sample_suboptimal_trajectory()
            true_pi = load_generated_policy(f"ICFMDPs/gridworld_policy__.jld2", 10, 16)
            simulated_pi = load_generated_policy(f"ICFMDPs/gridworld_policy_{max_transitions}_{delta}.jld2", 10, 16)

            # True CFIMDP
            cf_bound_calculator = MDPMultiStepCFBoundCalculator(mdp.transition_matrix)
            true_cf_imdp = cf_bound_calculator.calculate_bounds(observed_path)

            all_icfmdp_rewards, all_simulated_rewards = evaluate_policies(mdp.rewards, true_pi, simulated_pi, true_cf_imdp)

            all_icfmdp_rewards = np.array(all_icfmdp_rewards).reshape(2000000, 10)
            all_simulated_rewards = np.array(all_simulated_rewards).reshape(2000000, 10)

            mean_icfmdp_rewards = np.mean(np.array(all_icfmdp_rewards), axis=0)
            std_icfmdp_rewards = np.std(np.array(all_icfmdp_rewards), axis=0)
            mean_simulated_rewards = np.mean(np.array(all_simulated_rewards), axis=0)
            std_simulated_rewards = np.std(np.array(all_simulated_rewards), axis=0)

            upper_icfmdp_errors = np.clip(mean_icfmdp_rewards + std_icfmdp_rewards, None, 100) - mean_icfmdp_rewards
            upper_simulated_errors = np.clip(mean_simulated_rewards + std_simulated_rewards, None, 100) - mean_simulated_rewards
            lower_icfmdp_errors = mean_icfmdp_rewards - np.clip(mean_icfmdp_rewards - std_icfmdp_rewards, -100, None)
            lower_simulated_errors = mean_simulated_rewards - np.clip(mean_simulated_rewards - std_simulated_rewards, -100, None)

            with open(f"GridWorld Results.txt", "a") as file:
                file.write(f"Observed trajectory: {observed_path}\n\n")
                file.write(f"Hyperparameters = ({max_transitions, delta})\n\n")
                file.write(f"Average Results: \n\n")
                file.write(f"Mean ICFMDP rewards: {mean_icfmdp_rewards}\n")
                file.write(f"Upper bounds: {upper_icfmdp_errors}\n")
                file.write(f"Lower bounds: {lower_icfmdp_errors}\n")
                file.write(f"Mean simulated ICFMDP rewards: {mean_simulated_rewards}\n")
                file.write(f"Upper bounds: {upper_simulated_errors}\n")
                file.write(f"Lower bounds: {lower_simulated_errors}\n\n")

            all_rewards.append(mean_simulated_rewards)

        with open(f"gridworld_all_rewards.pickle", "wb") as f:
            pickle.dump(all_rewards, f)
        
        with open(f"gridworld_true_imdp_rewards.pickle", "wb") as f:
            pickle.dump(mean_icfmdp_rewards)


    elif function_name == "compare_bound_widths":
        def calculate_average_probability_width_ICFMDP(interval_CF_MDP):
            num_transitions = 10 * 16 * 4 * 16
            num_non_zero_transitions = 0
            total_prob_bounds = 0

            for t in range(10):
                for s in range(16):
                    for a in range(4):
                        for s_prime in range(16):
                            lb = interval_CF_MDP[t, s, a, s_prime, 0]
                            ub = interval_CF_MDP[t, s, a, s_prime, 1]

                            if not (ub == 0.0):
                                num_non_zero_transitions += 1
                                total_prob_bounds += ub - lb

            total_non_zero_prob_bounds = total_prob_bounds / num_non_zero_transitions
            total_prob_bounds /= num_transitions

            return total_prob_bounds, total_non_zero_prob_bounds
        
        def calculate_average_probability_width_IMDP(interval_CF_MDP):
            num_transitions = 16 * 4 * 16
            num_non_zero_transitions = 0
            total_prob_bounds = 0

            for s in range(16):
                for a in range(4):
                    for s_prime in range(16):
                        lb = interval_CF_MDP[s, a, s_prime, 0]
                        ub = interval_CF_MDP[s, a, s_prime, 1]

                        if not (ub == 0.0):
                            num_non_zero_transitions += 1
                            total_prob_bounds += ub - lb

            total_non_zero_prob_bounds = total_prob_bounds / num_non_zero_transitions
            total_prob_bounds /= num_transitions

            return total_prob_bounds, total_non_zero_prob_bounds

        with open(f"gridworld_bound_widths.txt", "a") as bounds_file:
            for (max_transitions, delta) in hyperparameters:
                with open(f"MDPs/simulated_gridworld_{max_transitions}_{delta}.pickle", "rb") as f:
                    simulated_cf_imdp = pickle.load(f)

                    probs, non_zero_probs = calculate_average_probability_width_IMDP(simulated_cf_imdp)

                    bounds_file.write(f"Hyperparameters = ({max_transitions}, {delta})\n")
                    bounds_file.write(f"Total probs = {probs}, Total non zero probs = {non_zero_probs}\n\n")

        with open(f"gridworld_cf_bound_widths.txt", "a") as bounds_file:
            for (max_transitions, delta) in hyperparameters:
                with open(f"MDPs/simulated_cf_gridworld_{max_transitions}_{delta}.pickle", "rb") as f:
                    simulated_cf_imdp = pickle.load(f)

                    probs, non_zero_probs = calculate_average_probability_width_ICFMDP(simulated_cf_imdp)

                    bounds_file.write(f"Hyperparameters = ({max_transitions}, {delta})\n")
                    bounds_file.write(f"Total probs = {probs}, Total non zero probs = {non_zero_probs}\n\n")



    elif function_name == "compare_policies":
        for i, (max_transitions, delta) in enumerate(hyperparameters):
            true_pi = load_generated_policy(f"ICFMDPs/gridworld_policy__.jld2", 10, 16)
            simulated_pi = load_generated_policy(f"ICFMDPs/gridworld_policy_{max_transitions}_{delta}.jld2", 10, 16)

            print(true_pi)
            print(simulated_pi)

            true_v = load_generated_policy(f"ICFMDPs/gridworld_value_pessimistic__.jld2", 10, 16)
            simulated_v = load_generated_policy(f"ICFMDPs/gridworld_value_pessimistic_{max_transitions}_{delta}.jld2", 10, 16)

            print(true_v)
            print(simulated_v)                 


    elif function_name == "plot_rewards":
        with open(f"gridworld_all_rewards.pickle", "rb") as f:
            mean_rewards = pickle.load(f)

        plt.figure(figsize=(8, 6))

        final_state_rewards = {}

        for i, (max_transitions, delta) in enumerate(hyperparameters):
            if str(delta) in final_state_rewards:
                final_state_rewards[str(delta)].append(mean_rewards[i][-1])

            else:
                final_state_rewards[str(delta)] = [mean_rewards[i][-1]]

        for delta in possible_deltas:
            y_vals = final_state_rewards[str(delta)]
            x_vals = possible_transitions

            plt.scatter(x_vals, y_vals, label=f"δ =  {delta}")

        plt.xlabel("Number of Transitions")
        plt.ylabel("Average Reward")
        plt.title("Average Reward vs. Number of Transitions")
        plt.legend(title="δ")
        plt.grid(True)

        plt.savefig("gridworld_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

    else:
       print(f"Function '{function_name}' is not recognized.")

main()
