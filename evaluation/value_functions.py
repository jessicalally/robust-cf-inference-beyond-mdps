import numpy as np
from scipy.optimize import linprog
from environments import Model_Checking_MDP

def get_state_reward_model_checking(mdp, s):
    assert(isinstance(mdp, Model_Checking_MDP))

    if s in mdp.goal_states:
        return 1.0
    elif s in mdp.collision_states:
        return -1.0
    
    return 0.0

# Evaluate the value function for a MDP under a given policy.
def value_iteration(mdp, policy, max_t):
    V = np.zeros((max_t+1, mdp.n_states))

    for t in reversed(range(max_t)):
        for s in range(mdp.n_states):
            a = policy[s]

            if isinstance(mdp, Model_Checking_MDP):
                V[t, s] = get_state_reward_model_checking(mdp, s) + np.dot(mdp.transition_matrix[s, a], V[t+1])
            else:
                V[t, s] = mdp.rewards[s, a] + np.dot(mdp.transition_matrix[s, a], V[t+1])
    return V


def value_iteration_under_policy(transition_matrix, mdp, policy, max_t):
    V = np.zeros((max_t+1, mdp.n_states))

    for t in reversed(range(max_t)):
        for s in range(mdp.n_states):
            a = policy[t, s]

            if isinstance(mdp, Model_Checking_MDP):
                V[t, s] = get_state_reward_model_checking(mdp, s) + np.dot(transition_matrix[t, s, a], V[t+1])
            else:
                V[t, s] = mdp.rewards[s, a] + np.dot(transition_matrix[t, s, a], V[t+1])
    return V


# Evaluate the value function for a non-stationary MDP under a given policy.
def value_iteration_non_stationary(transition_matrix, mdp, policy, max_t):
    V = np.zeros((max_t+1, mdp.n_states))

    for t in reversed(range(max_t)):
        for s in range(mdp.n_states):
            a = policy[s]

            if isinstance(mdp, Model_Checking_MDP):
                V[t, s] = get_state_reward_model_checking(mdp, s) + np.dot(transition_matrix[t, s, a], V[t+1])
            else:
                V[t, s] = mdp.rewards[s, a] + np.dot(transition_matrix[t, s, a], V[t+1])

    return V


def policy_iteration_non_stationary(transition_matrix, mdp, max_t):
    V = np.zeros((max_t+1, mdp.n_states))
    policy = np.zeros((max_t, mdp.n_states), dtype=int)

    for s in range(mdp.n_states):
        V[max_t, s] = mdp.rewards[s, 0]

    for t in reversed(range(max_t)):
        for s in range(mdp.n_states):
            Q_s = np.zeros(mdp.n_actions)

            for a in range(mdp.n_actions):
                if isinstance(mdp, Model_Checking_MDP):
                    Q_s[a] = get_state_reward_model_checking(mdp, s) + np.dot(transition_matrix[t, s, a], V[t+1])
                else:
                    Q_s[a] = mdp.rewards[s, a] + np.dot(transition_matrix[t, s, a], V[t+1])

            # Best action and value
            policy[t, s] = np.argmax(Q_s)
            V[t, s] = Q_s[policy[t, s]]

    return V, policy


# Evaluate the value function for an IMDP under a given policy.
def value_iteration_imdp(imdp, mdp, policy, max_t, gamma=1.0):
    V_low = np.zeros((max_t+1, mdp.n_states))
    V_high = np.zeros((max_t+1, mdp.n_states))

    for t in reversed(range(max_t)):
        for s in range(mdp.n_states):
            a = policy[s]

            # Transition probabilities
            P_low = imdp[t, s, a, :, 0]
            P_high = imdp[t, s, a, :, 1]

            # --- Worst-case LP ---
            c_min = V_low[t+1]  # objective: minimise sum p * V_next
            A_eq = np.ones((1, mdp.n_states))
            b_eq = np.array([1.0])

            bounds = [(P_low[i], P_high[i]) for i in range(mdp.n_states)]

            res_min = linprog(c=c_min, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if not res_min.success:
                print(f"LP failed at t={t}, s={s}, a={a}")
                print(f"P_low={np.nonzero(P_low)} {P_low[np.nonzero(P_low)[0]]}")
                print(f"P_high={np.nonzero(P_high)} {P_high[np.nonzero(P_high)[0]]}")
                print(res_min.message)

            if isinstance(mdp, Model_Checking_MDP):
                V_low[t, s] = get_state_reward_model_checking(mdp, s) + gamma * res_min.fun
            else:
                V_low[t, s] = mdp.rewards[s, a] + gamma * res_min.fun

            # --- Best-case LP ---
            c_max = -V_high[t+1]  # maximise sum p * V_next = minimise -sum p * V_next
            res_max = linprog(c=c_max, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if isinstance(mdp, Model_Checking_MDP):
                V_high[t, s] = get_state_reward_model_checking(mdp, s) - gamma * res_max.fun
            else:
                V_high[t, s] = mdp.rewards[s, a] - gamma * res_max.fun
    
    return V_low, V_high

def policy_eval_imdp_deterministic_max_actions(observed_path, imdp, mdp, policies, max_t, gamma=1.0, tol=1e-12):
    V_low = np.zeros((max_t+1, max_t+1, mdp.n_states))
    observed_actions = observed_path[:, 2]

    for k in range(max_t):
        for t in reversed(range(max_t)):
            for s in range(mdp.n_states):
                a = policies[k, t, s]   # deterministic action choice

                if k == 0:
                    assert(a == observed_actions[t])

                P_low = imdp[t, s, a, :, 0]
                P_high = imdp[t, s, a, :, 1]

                A_eq = np.ones((1, mdp.n_states))
                b_eq = np.array([1.0])
                bounds = [(P_low[i], P_high[i]) for i in range(mdp.n_states)]

                # --- Worst-case (minimizing next value) ---
                if np.allclose(P_low, P_high, atol=tol) and np.allclose(np.sum(P_low), 1.0, atol=tol):
                    if a == observed_actions[t]:
                        expected_low = np.dot(P_low, V_low[k, t+1])
                    else:
                        expected_low = np.dot(P_low, V_low[k-1, t+1])

                    if isinstance(mdp, Model_Checking_MDP):
                        V_low[k, t, s] = get_state_reward_model_checking(mdp, s) + gamma * expected_low
                    else:
                        V_low[k, t, s] = mdp.rewards[s, a] + gamma * expected_low

                else:
                    if a == observed_actions[t]:
                        c_min = V_low[k, t+1]  # minimise p·V_next
                    else:
                        c_min = V_low[k-1, t+1]

                    res_min = linprog(c=c_min, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
                    if not res_min.success:
                        raise RuntimeError(f"LP failed (LOW) at t={t}, s={s}: {res_min.message}")
                    
                    if isinstance(mdp, Model_Checking_MDP):
                        V_low[k, t, s] = get_state_reward_model_checking(mdp, s) + gamma * res_min.fun
                    else:
                        V_low[k, t, s] = mdp.rewards[s, a] + gamma * res_min.fun

    return V_low



def policy_eval_imdp_deterministic(imdp, mdp, policy, max_t, gamma=1.0, tol=1e-9):
    V_low = np.zeros((max_t+1, mdp.n_states))
    V_high = np.zeros((max_t+1, mdp.n_states))

    worst_case_CFMDP = np.zeros((max_t, mdp.n_states, mdp.n_states))
    best_case_CFMDP = np.zeros((max_t, mdp.n_states, mdp.n_states))

    for s in range(mdp.n_states):
        if isinstance(mdp, Model_Checking_MDP):
            V_low[max_t, s] = get_state_reward_model_checking(mdp, s)
            V_high[max_t, s] = get_state_reward_model_checking(mdp, s)
        else:
            V_low[max_t, s] = mdp.rewards[s, 0]
            V_high[max_t, s] = mdp.rewards[s, 0] # rewards same for all actions

    for t in reversed(range(max_t)):
        for s in range(mdp.n_states):
            a = policy[t, s]   # deterministic action choice

            P_low = imdp[t, s, a, :, 0]
            P_high = imdp[t, s, a, :, 1]

            A_eq = np.ones((1, mdp.n_states))
            b_eq = np.array([1.0])
            bounds = [(P_low[i], P_high[i]) for i in range(mdp.n_states)]

            # --- Worst-case (minimizing next value) ---
            if np.allclose(P_low, P_high, atol=tol) and np.allclose(np.sum(P_low), 1.0, atol=tol):
                expected_low = np.dot(P_low, V_low[t+1])

                if isinstance(mdp, Model_Checking_MDP):
                    V_low[t, s] = get_state_reward_model_checking(mdp, s) + gamma * expected_low
                else:
                    V_low[t, s] = mdp.rewards[s, a] + gamma * expected_low
                worst_case_CFMDP[t, s] = P_low

            else:
                c_min = V_low[t+1]  # minimise p·V_next
                res_min = linprog(c=c_min, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
                if not res_min.success:
                    raise RuntimeError(f"LP failed (LOW) at t={t}, s={s}: {res_min.message}")
                
                if isinstance(mdp, Model_Checking_MDP):
                    V_low[t, s] = get_state_reward_model_checking(mdp, s) + gamma * res_min.fun
                else:
                    V_low[t, s] = mdp.rewards[s, a] + gamma * res_min.fun
                worst_case_CFMDP[t, s] = res_min.x

            # --- Best-case (maximizing next value) ---
            if np.allclose(P_low, P_high, atol=tol) and np.allclose(np.sum(P_high), 1.0, atol=tol):
                expected_high = np.dot(P_high, V_high[t+1])

                if isinstance(mdp, Model_Checking_MDP):
                    V_high[t, s] = get_state_reward_model_checking(mdp, s) + gamma * expected_high
                else:
                    V_high[t, s] = mdp.rewards[s, a] + gamma * expected_high

                best_case_CFMDP[t, s] = P_high
            else:
                c_max = -V_high[t+1]  # maximise = minimise negative
                res_max = linprog(c=c_max, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
                if not res_max.success:
                    raise RuntimeError(f"LP failed (HIGH) at t={t}, s={s}: {res_max.message}")
                
                if isinstance(mdp, Model_Checking_MDP):
                    V_high[t, s] = get_state_reward_model_checking(mdp, s) - gamma * res_max.fun
                else:
                    V_high[t, s] = mdp.rewards[s, a] - gamma * res_max.fun

                best_case_CFMDP[t, s] = res_max.x

    return V_low, V_high, worst_case_CFMDP, best_case_CFMDP


def action_value_iteration_imdp(imdp, mdp, max_t, gamma=1.0, tol=1e-9):
    Q_low = np.zeros((max_t+1, mdp.n_states, mdp.n_actions))
    Q_high = np.zeros((max_t+1, mdp.n_states, mdp.n_actions))

    # Identify the worst- and best-case MDPs, under the target policy.
    worst_case_CFMDP = np.zeros(shape=(max_t, mdp.n_states, mdp.n_actions, mdp.n_states))
    best_case_CFMDP = np.zeros(shape=(max_t, mdp.n_states, mdp.n_actions, mdp.n_states))

    for s in range(mdp.n_states):
        for a in range(mdp.n_actions):
            if isinstance(mdp, Model_Checking_MDP):
                Q_low[max_t, s, a] = get_state_reward_model_checking(mdp, s)
                Q_high[max_t, s, a] = get_state_reward_model_checking(mdp, s)
            else:
                Q_low[max_t, s, a] = mdp.rewards[s, a]
                Q_high[max_t, s, a] = mdp.rewards[s, a]

    for t in reversed(range(max_t)):
        for s in range(mdp.n_states):
            for a in range(mdp.n_actions):
                # Transition probabilities (intervals)
                P_low = imdp[t, s, a, :, 0]
                P_high = imdp[t, s, a, :, 1]

                # Equality constraint: sum of probabilities must equal 1
                A_eq = np.ones((1, mdp.n_states))
                b_eq = np.array([1.0])

                bounds = [(P_low[i], P_high[i]) for i in range(mdp.n_states)]

                # --- Worst-case (lower bound) ---
                if np.allclose(P_low, P_high, atol=tol) and np.allclose(np.sum(P_low), 1.0, atol=tol):
                    # Degenerate case: unique probability distribution
                    expected_low = np.dot(P_low, Q_low[t+1].max(axis=1))

                    if isinstance(mdp, Model_Checking_MDP):
                        Q_low[t, s, a] = get_state_reward_model_checking(mdp, s) + (gamma * expected_low)
                    else:
                        Q_low[t, s, a] = mdp.rewards[s, a] + (gamma * expected_low)
                    worst_case_CFMDP[t, s, a] = P_low
                else:
                    c_min = Q_low[t+1].max(axis=1)  # objective: minmise sum p * V_next
                    res_min = linprog(c=c_min, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

                    if not res_min.success:
                        print(f"LP failed (LOW) at t={t}, s={s}, a={a}")
                        print(f"P_low={P_low}, P_high={P_high}")
                        print(res_min.message)

                    if isinstance(mdp, Model_Checking_MDP):
                        Q_low[t, s, a] = get_state_reward_model_checking(mdp, s) + (gamma * res_min.fun)
                    else:
                        Q_low[t, s, a] = mdp.rewards[s, a] + (gamma * res_min.fun)

                    worst_case_CFMDP[t, s, a] = res_min.x

                # --- Best-case (upper bound) ---
                if np.allclose(P_low, P_high, atol=tol) and np.allclose(np.sum(P_high), 1.0, atol=tol):
                    # Degenerate case: unique probability distribution
                    expected_high = np.dot(P_high, Q_high[t+1].max(axis=1))

                    if isinstance(mdp, Model_Checking_MDP):
                        Q_high[t, s, a] = get_state_reward_model_checking(mdp, s) + (gamma * expected_high)
                    else:
                        Q_high[t, s, a] = mdp.rewards[s, a] + (gamma * expected_high)
                    best_case_CFMDP[t, s, a] = P_high
                else:
                    c_max = -Q_high[t+1].max(axis=1)  # maximise = minmise negative
                    res_max = linprog(c=c_max, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

                    if not res_max.success:
                        print(f"LP failed (HIGH) at t={t}, s={s}, a={a}")
                        print(f"P_low={P_low}, P_high={P_high}")
                        print(res_max.message)

                    if isinstance(mdp, Model_Checking_MDP):
                        Q_high[t, s, a] = get_state_reward_model_checking(mdp, s) - (gamma * res_max.fun)
                    else:
                        Q_high[t, s, a] = mdp.rewards[s, a] - (gamma * res_max.fun)

                    best_case_CFMDP[t, s, a] = res_max.x

    return Q_low, Q_high, worst_case_CFMDP, best_case_CFMDP


def value_iteration_imdp(imdp, mdp, policy, max_t, gamma=1.0, tol=1e-9):
    V_low = np.zeros((max_t+1, mdp.n_states))
    V_high = np.zeros((max_t+1, mdp.n_states))

    for t in reversed(range(max_t)):
        for s in range(mdp.n_states):
            a = policy[s]

            # Transition probabilities (intervals)
            P_low = imdp[t, s, a, :, 0]
            P_high = imdp[t, s, a, :, 1]

            # Equality constraint: sum of probabilities must equal 1
            A_eq = np.ones((1, mdp.n_states))
            b_eq = np.array([1.0])

            # Bounds
            bounds = [(P_low[i], P_high[i]) for i in range(mdp.n_states)]

            # --- Worst-case (lower bound) ---
            if np.allclose(P_low, P_high, atol=tol) and np.allclose(np.sum(P_low), 1.0, atol=tol):
                # Degenerate case: unique probability distribution
                expected_low = np.dot(P_low, V_low[t+1])

                if isinstance(mdp, Model_Checking_MDP):
                    V_low[t, s] = get_state_reward_model_checking(mdp, s) + (gamma * expected_low)
                else:
                    V_low[t, s] = mdp.rewards[s, a] + (gamma * expected_low)
            else:
                c_min = V_low[t+1]  # objective: minimise sum p * V_next
                res_min = linprog(c=c_min, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                
                if not res_min.success:
                    print(f"LP failed (LOW) at t={t}, s={s}, a={a}")
                    print(f"P_low={P_low}, P_high={P_high}")
                    print(res_min.message)
                
                if isinstance(mdp, Model_Checking_MDP):
                    V_low[t, s] = get_state_reward_model_checking(mdp, s) + (gamma * res_min.fun)
                else:
                    V_low[t, s] = mdp.rewards[s, a] + (gamma * res_min.fun)

            # --- Best-case (upper bound) ---
            if np.allclose(P_low, P_high, atol=tol) and np.allclose(np.sum(P_high), 1.0, atol=tol):
                # Degenerate case: unique probability distribution
                expected_high = np.dot(P_high, V_high[t+1])

                if isinstance(mdp, Model_Checking_MDP):
                    V_high[t, s] = get_state_reward_model_checking(mdp, s) + (gamma * expected_high)
                else:
                    V_high[t, s] = mdp.rewards[s, a] + (gamma * expected_high)
           
            else:
                c_max = -V_high[t+1]  # maximise = minimise negative
                res_max = linprog(c=c_max, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
                
                if not res_max.success:
                    print(f"LP failed (HIGH) at t={t}, s={s}, a={a}")
                    print(f"P_low={P_low}, P_high={P_high}")
                    print(res_max.message)

                if isinstance(mdp, Model_Checking_MDP):
                    V_high[t, s] = get_state_reward_model_checking(mdp, s) - (gamma * res_max.fun)
                else:
                    V_high[t, s] = mdp.rewards[s, a] - (gamma * res_max.fun)

    return V_low, V_high