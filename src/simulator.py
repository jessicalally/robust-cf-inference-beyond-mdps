# Simulates transitions from an MDP.
from collections import defaultdict
from math import sqrt, log
import numpy as np
from environments import Model_Checking_MDP
from tqdm import trange
from multiprocessing import Pool, cpu_count
import random
from scipy.stats import binomtest

class Simulator:

    def __init__(self, mdp, num_episodes, max_steps, epsilon, seed=None, init_state=None):
        self.mdp = mdp
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.possible_transitions = mdp.get_possible_transitions()
        self.seed = seed
        self.init_state = init_state
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)


    def _simulate_transitions(self):
        transitions = []
        trajectories = []

        for e in trange(self.num_episodes):

            if self.init_state is None:
                # Start from random non-terminal state.
                state = np.random.choice(np.where(self.mdp.sink_state == False)[0])
            else:
                # Or, start from defined state.
                state = self.init_state

            traj = []
            
            for _ in range(self.max_steps):
                # Pick from valid actions for the current state.
                valid_actions = np.where(self.mdp.valid_actions[state])[0]
                action = np.random.choice(valid_actions)
                is_done, next_state = self.mdp.sample_next_state(state, action)
                transitions.append((state, action, next_state))

                if isinstance(self.mdp, Model_Checking_MDP):
                    if state in self.mdp.goal_states:
                        traj.append((state, action, next_state, 1.0))
                    elif state in self.mdp.collision_states:
                        traj.append((state, action, next_state, -1.0))
                    else:
                        traj.append((state, action, next_state, 0.0))
                else:
                    traj.append((state, action, next_state, self.mdp.rewards[state, action]))

                if is_done:
                    # If we are in a sink state, there is no point in continuing to sample the trajectory.
                    break
                
                state = next_state

            trajectories.append(traj)

        return transitions, trajectories    

    def learn_imdp(self):
        state_action_counts = defaultdict(int)
        transition_counts = defaultdict(int)

        observed_transitions = self._simulate_transitions()

        for s, a, s_next in observed_transitions:
            state_action_counts[(s, a)] += 1
            transition_counts[(s, a, s_next)] += 1

        imdp = np.zeros((self.mdp.n_states, self.mdp.n_actions, self.mdp.n_states, 2))  # [LB, UB]

        for (s, a, s_prime) in self.possible_transitions:
            N = state_action_counts.get((s, a), 0)
            count = transition_counts.get((s, a, s_prime), 0)
            if N > 0:
                p_hat = count / N
                # Hoeffding-style bound
                epsilon = sqrt(log(2 / self.gamma) / (2 * N))
                lb = max(self.epsilon, p_hat - epsilon)
                ub = min(1.0, p_hat + epsilon)
            else:
                lb, ub = self.epsilon, 1.0  # No data: full uncertainty

            imdp[s, a, s_prime, 0] = lb
            imdp[s, a, s_prime, 1] = ub

        return imdp
    

    def get_counts(self, prev_state_action_counts, prev_transition_counts):
        # Carry over counts from previous 10^N-1 samples.
        state_action_counts = prev_state_action_counts
        transition_counts = prev_transition_counts

        # Generate simulations.
        observed_transitions, trajectories = self._simulate_transitions()
        possible_next_states = {}

        # Update counts from observed transitions.
        for s, a, s_next in observed_transitions:
            state_action_counts[(s, a)] += 1
            transition_counts[(s, a, s_next)] += 1

            if (s,a) not in possible_next_states:
                possible_next_states[(s,a)] = set()
            possible_next_states[(s,a)].add(s_next)

        scores = []

        # We calculate the log probability of a particular transition, rather than the prob, to avoid dealing with very small floating points.
        # Then, we use Laplace smoothing to avoid -∞ log probability.
        def log_prob_transition(s,a,s_next,alpha=1.0):
            num = transition_counts[(s,a,s_next)] + alpha
            denom = sum(transition_counts[(s,a,x)] + alpha 
                        for x in possible_next_states[(s,a)])
            return np.log(num / denom)

        # We compute a score for each trajectory by averaging the log-probability of its transitions.
        for traj in trajectories:
            logs = [log_prob_transition(s,a,s_prime) for (s,a,s_prime,_) in traj]
            scores.append(np.mean(logs))

        # for score in scores:
        #     print(f"Score: {score}")

        # Then, we compute the 75th percentile of trajectory scores.
        q75 = np.percentile(scores, 75)

        # Finally, we take a suboptimal path (i.e., a path that does not achieve the max reward) from the 75th percentile.
        if isinstance(self.mdp, Model_Checking_MDP):
            likely_suboptimal_paths = [traj for traj, score in zip(trajectories, scores) if score >= q75 and not 1.0 in traj[:][-1]]
        else:
            likely_suboptimal_paths = [traj for traj, score in zip(trajectories, scores) if score >= q75 and not traj[-1][-1] == self.mdp.max_reward]

        return state_action_counts, transition_counts, likely_suboptimal_paths[0]


    def learn_imdp_pac_guaranteed(self, state_action_counts, transition_counts, gamma, method="hoeffding"):
        total_possible_nondeterministic_transitions = len(self.mdp.get_possible_nondeterministic_transitions())
        gamma_p = gamma / total_possible_nondeterministic_transitions

        imdp = np.zeros((self.mdp.n_states, self.mdp.n_actions, self.mdp.n_states, 2))  # [LB, UB]

        for (s, a, s_prime) in self.possible_transitions:
            N = state_action_counts.get((s, a), 0)
            count = transition_counts.get((s, a, s_prime), 0)

            next_state_probs = self.mdp.transition_matrix[s, a]
            num_next_states = np.count_nonzero(next_state_probs)

            assert(num_next_states > 0)

            if num_next_states == 1:
                # This is the only possible next state, so should have [1.0, 1.0] bounds.
                lb = 1.0
                ub = 1.0
            else:
                if N > 0:
                    p_hat = count / N

                    if method == "hoeffding":
                        # Hoeffding-style bound
                        delta_p = sqrt(log(2 / gamma_p) / (2 * N))
                        lb = max(self.epsilon, p_hat - delta_p)
                        ub = min(1.0, p_hat + delta_p)

                    elif method == "clopper-pearson":
                        result = binomtest(k=count, n=N)
                        lb, ub = result.proportion_ci(confidence_level=1-gamma_p, method='exact')
                        lb = max(self.epsilon, lb)
                        lb = min(lb, 1.0)
                        ub = max(self.epsilon, ub)
                        ub = min(1.0, ub)

                        # Check correctness of bounds.
                        assert(lb <= 1.0)
                        assert(ub >= 0.0)
                        assert(ub >= lb)
                    else:
                        raise ValueError("Invalid PAC method, options are: hoeffding, clopper-pearson") 
                else:
                    lb, ub = self.epsilon, 1.0  # No data: full uncertainty

            imdp[s, a, s_prime, 0] = lb
            imdp[s, a, s_prime, 1] = ub

        return imdp

