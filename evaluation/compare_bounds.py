import numpy as np

def compare_approx_vs_tight_bounds_mdp(approx_simulated_cf_imdp, tight_simulated_cf_imdp):
    width_diffs_this_mdp = []
    lb_diffs_this_mdp = []
    non_zero_lb_diffs_this_mdp = []
    num_nontight_approx_this_mdp = 0
    num_nonzero_transitions_this_mdp = 0

    n_timesteps = approx_simulated_cf_imdp.shape[0]
    n_states = approx_simulated_cf_imdp.shape[1]
    n_actions = approx_simulated_cf_imdp.shape[2]

    for t in range(n_timesteps):
        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    # Assert UBs are the same.
                    assert(approx_simulated_cf_imdp[t, s, a, s_prime, 1] == tight_simulated_cf_imdp[t, s, a, s_prime, 1])

                    if approx_simulated_cf_imdp[t, s, a, s_prime, 1] > 0:
                        num_nonzero_transitions_this_mdp += 1

                    # Compare the bound widths.
                    approx_width = approx_simulated_cf_imdp[t, s, a, s_prime, 1] - approx_simulated_cf_imdp[t, s, a, s_prime, 0]
                    tight_width = tight_simulated_cf_imdp[t, s, a, s_prime, 1] - tight_simulated_cf_imdp[t, s, a, s_prime, 0]
                    width_diff = abs(approx_width - tight_width)
                    width_diffs_this_mdp.append(width_diff)

                    # Compare the lower bounds.
                    approx_lb = approx_simulated_cf_imdp[t, s, a, s_prime, 0]
                    tight_lb = tight_simulated_cf_imdp[t, s, a, s_prime, 0]

                    lb_diff = abs(tight_lb - approx_lb)
                    lb_diffs_this_mdp.append(lb_diff)

                    if not tight_lb == approx_lb:
                        non_zero_lb_diffs_this_mdp.append(lb_diff)
                        num_nontight_approx_this_mdp += 1

    average_width_diff_this_mdp = np.mean(np.array(width_diffs_this_mdp))
    average_lb_diff_this_mdp = np.mean(np.array(lb_diffs_this_mdp))
    average_non_zero_diff_this_mdp = np.mean(np.array(non_zero_lb_diffs_this_mdp))
    total_percentage_affected_this_mdp = num_nontight_approx_this_mdp / (t * s * a * s)
    approx_non_zero_affected_this_mdp = num_nontight_approx_this_mdp / num_nonzero_transitions_this_mdp

    return average_width_diff_this_mdp, average_lb_diff_this_mdp, average_non_zero_diff_this_mdp, total_percentage_affected_this_mdp, approx_non_zero_affected_this_mdp


def compare_approx_vs_tight_bounds(env_name, hyperparameters, sampled_paths, simulated_cf_imdps_approx, simulated_cf_imdps_tight):
    print(f"Comparing approx vs tight...")
    for j, (max_transitions, delta) in enumerate(hyperparameters):
        width_diffs = []
        lb_diffs = []
        non_zero_lb_diffs = []
        total_percentage_affected = []
        approx_non_zero_affected = []

        for i, _ in enumerate(sampled_paths):
            average_width_diff_this_mdp, average_lb_diff_this_mdp, average_non_zero_diff_this_mdp, total_percentage_affected_this_mdp, approx_non_zero_affected_this_mdp = compare_approx_vs_tight_bounds_mdp(simulated_cf_imdps_approx[i, j], simulated_cf_imdps_tight[i, j])

            width_diffs.append(average_width_diff_this_mdp)
            lb_diffs.append(average_lb_diff_this_mdp)
            non_zero_lb_diffs.append(average_non_zero_diff_this_mdp)
            total_percentage_affected.append(total_percentage_affected_this_mdp)
            approx_non_zero_affected.append(approx_non_zero_affected_this_mdp)

        average_width_diff = np.mean(np.array(width_diffs))
        average_lb_diff = np.mean(np.array(lb_diffs))
        average_non_zero_diff = np.mean(np.array(non_zero_lb_diffs))
        total_percentage_affected = np.mean(np.array(total_percentage_affected))
        approx_non_zero_affected = np.mean(np.array(approx_non_zero_affected))
        
        with open(f"results/{env_name}/pac/{env_name}_approx_vs_tight_max_actions.txt", "a") as f:
            f.write(f"## HYPERPARAMETERS = {max_transitions}, {delta} ##\n\n")
            f.write(f"Percentage affected = {total_percentage_affected * 100}%\n")
            f.write(f"Percentage nonzero transitions affected = {approx_non_zero_affected * 100}%\n")
            f.write(f"Average width diff = {average_width_diff}\n")
            f.write(f"Average LB diff = {average_lb_diff}\n")
            f.write(f"Average non-zero LB diff = {average_non_zero_diff}\n")
