from collections import defaultdict
from itertools import product
import numpy as np
from evaluation.path_sampling import *
from evaluation.value_functions import *
from src import simulate_counts, learn_imdp_pac_guaranteed
from src import ParallelMultiStepCFBoundCalculatorApprox as IMDPMultiStepCFBoundCalculatorApprox
from src import MultiStepCFBoundCalculator as MDPMultiStepCFBoundCalculator

def check_cf_correctness(mdp, max_t, possible_trajectories=[100000], possible_gammas=[0.1]):
    np.random.seed(1)

    previously_sampled_paths = None
    print(f"init state = {mdp.init_state}")

    # Generate IMDP from n_runs.
    state_action_counts = defaultdict(int)
    transition_counts = defaultdict(int)

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

    # Take deterministic suboptimal behaviour policy.
    behaviour_policy_random = mdp.policy_with_randomization(mdp.optimal_policy, 0.5)
    
    # Generate suboptimal behaviour policy.
    if behaviour_policy_random.shape == (mdp.n_states, ):
        behaviour_policy = behaviour_policy_random
    else:
        behaviour_policy = np.zeros(shape=(mdp.n_states), dtype=int)

        for s in range(mdp.n_states):
            behaviour_policy[s] = np.random.choice(mdp.n_actions, p=behaviour_policy_random[s])
    
    # Take target policy (in this case, optimal policy).
    target_policy = mdp.optimal_policy
    
    # Evaluate expected reward under target policy across true MDP.
    interventional_value_function = value_iteration(mdp, target_policy, max_t)
    interventional_expected_reward = interventional_value_function[0, mdp.init_state]

    runs_and_gamma = list(product(possible_trajectories, possible_gammas))

    V_lows = []
    V_highs = []
    V_lows_simulated_all = {}
    V_highs_simulated_all = {}

    for n_runs, gamma in runs_and_gamma:
        V_lows_simulated_all[(n_runs, gamma)] = []
        V_highs_simulated_all[(n_runs, gamma)] = []
    
    prev_n_paths = None

    for n_paths in [1, 10, 20, 40, 60, 80, 100]:
        print(f"n_paths={n_paths}")

        # Sample paths in the MDP under the behaviour policy.
        if previously_sampled_paths is None:
            sampled_paths = sample_paths(mdp, behaviour_policy, max_t, n_paths=n_paths)
        else:
            sampled_paths = sample_paths(mdp, behaviour_policy, max_t, n_paths=n_paths-prev_n_paths)

        prev_n_paths = n_paths

        # For each sampled path under behaviour policy, evaluate the CFIMDP, and the value functions.
        for _, sampled_path in enumerate(sampled_paths):
            cf_bound_calculator = MDPMultiStepCFBoundCalculator(mdp, mdp.transition_matrix)
            true_cf_imdp = cf_bound_calculator.calculate_bounds(sampled_path)

            # Evaluate the target policy on the true CFIMDP.
            V_low, V_high = value_iteration_imdp(true_cf_imdp, mdp, target_policy, max_t)

            V_lows.append(V_low[0, mdp.init_state])
            V_highs.append(V_high[0, mdp.init_state])

            for i, simulated_imdp in enumerate(simulated_imdps):
                n_runs, gamma = runs_and_gamma[i]

                # Compute the simulated CFIMDP and evaluate the target policy on this.
                cf_bound_calculator = IMDPMultiStepCFBoundCalculatorApprox(mdp, simulated_imdp)
                simulated_cf_imdp = cf_bound_calculator.calculate_bounds(sampled_path, n_jobs=64)

                V_low, V_high = value_iteration_imdp(simulated_cf_imdp, mdp, target_policy, max_t)
                V_lows_simulated_all[(n_runs, gamma)].append(V_low[0, mdp.init_state])
                V_highs_simulated_all[(n_runs, gamma)].append(V_high[0, mdp.init_state])

        # Compute mean (expected) reward
        pessimistic_expected_reward = np.mean(V_lows)
        optimistic_expected_reward = np.mean(V_highs)

        # Compute standard deviation.
        pessimistic_std = np.std(V_lows)
        optimistic_std = np.std(V_highs)

        print(f"Expected reward bounds = {pessimistic_expected_reward} +- {pessimistic_std}, {optimistic_expected_reward} +- {optimistic_std}")

        for n_runs, gamma in runs_and_gamma:
            print(f"({n_runs}, {gamma})")
            pessimistic_expected_reward_simulated = np.mean(V_lows_simulated_all[((n_runs, gamma))])
            optimistic_expected_reward_simulated = np.mean(V_highs_simulated_all[(n_runs, gamma)])
            pessimistic_std_simulated = np.std(V_lows_simulated_all[(n_runs, gamma)])
            optimistic_std_simulated = np.std(V_highs_simulated_all[(n_runs, gamma)])
        
            print(f"Expected reward bounds simulated = {pessimistic_expected_reward_simulated} += {pessimistic_std_simulated}, {optimistic_expected_reward_simulated} +- {optimistic_std_simulated}")
            print(f"True expected reward = {interventional_expected_reward}")
