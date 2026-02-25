import itertools
import math
import multiprocessing as mp
from multiprocessing import Process, Manager
import numpy as np

class GumbelMaxCFBoundCalculatorMDP(object):
    def __init__(self, mdp, n_states, n_actions):
        self.mdp = mdp
        self.n_states = n_states
        self.n_actions = n_actions
        np.seterr(divide='ignore', invalid='ignore')


    def truncated_gumbel(self, logit, truncation):
        assert not np.isneginf(logit)

        gumbel = np.random.gumbel(size=(truncation.shape[0])) + logit
        trunc_g = -np.log(np.exp(-gumbel) + np.exp(-truncation))
        return trunc_g


    def topdown(self, obs_logits, obs_state, nsamp=1):
        poss_next_states = obs_logits.shape[0]
        gumbels = np.zeros((nsamp, poss_next_states))

        # Sample top gumbels.
        topgumbel = np.random.gumbel(size=(nsamp))

        for next_state in range(poss_next_states):
            # This is the observed next state.
            if (next_state == obs_state) and not(np.isneginf(obs_logits[next_state])):
                gumbels[:, obs_state] = topgumbel - obs_logits[next_state]

            # These were the other feasible options (p > 0).
            elif not(np.isneginf(obs_logits[next_state])):
                gumbels[:, next_state] = self.truncated_gumbel(obs_logits[next_state], topgumbel) - obs_logits[next_state]

            # These had zero probability to start with, so are unconstrained.
            else:
                gumbels[:, next_state] = np.random.gumbel(size=nsamp)

        return gumbels

    
    def cf_posterior(self, obs_prob, intrv_prob, state, n_mc):
        obs_logits = np.log(obs_prob)
        next_state = state
        intrv_logits = np.log(intrv_prob)
        gumbels = self.topdown(obs_logits, next_state, n_mc)
        posterior = intrv_logits + gumbels
        intrv_posterior = np.argmax(posterior, axis=1)
        posterior_prob = np.zeros(np.size(intrv_prob, 0))
        
        for i in range(np.size(intrv_prob, 0)):
            posterior_prob[i] = np.sum(intrv_posterior == i) / n_mc

        return posterior_prob, intrv_posterior


    def cf_sample_prob(self, trajectories, all_actions, T, n_cf_samps=1): 
        n_mc = 10000

        P_cf = np.zeros(shape=(T, self.n_states, self.n_actions, self.n_states))
        
        for a in range(all_actions):
            for t in range(T):
                for _ in range(n_cf_samps):
                    obs_state = trajectories[t]
                    obs_current_state = int(obs_state[0])
                    obs_next_state = int(obs_state[1])
                    obs_action = int(obs_state[2])

                    for s in range(self.n_states):
                        obs_intrv = self.mdp[obs_current_state, obs_action, :]
                        cf_intrv = self.mdp[s, a, :]
                        cf_prob, s_p = self.cf_posterior(obs_intrv, cf_intrv, obs_next_state, n_mc)
                        
                        for s_p in range(len(cf_prob)):
                            P_cf[t, s, a, s_p] = cf_prob[s_p]

        return P_cf


    def cf_sample_prob_parallel(self, trajectory, a, time_idx, P_cf_save, n_cf_samps=1): 
        n_mc = 10000

        P_cf = {}

        for _ in range(n_cf_samps):
            obs_state = trajectory[time_idx]
            obs_current_state = int(obs_state[0])
            obs_next_state = int(obs_state[1])
            obs_action = int(obs_state[2])

            P_cf[a, time_idx] = np.zeros((int(self.n_states),int(self.n_states)))

            for s in range(self.n_states):
                obs_intrv = self.mdp[obs_current_state, obs_action, :]
                cf_intrv = self.mdp[s, a, :]
                cf_prob, _ = self.cf_posterior(obs_intrv, cf_intrv, obs_next_state, n_mc)
                P_cf[a,time_idx][s, :] = cf_prob

        P_cf_save[(a,time_idx)] = P_cf

    
    def run_gumbel_sampling_single_threaded(self, trajectories):
        n_steps = trajectories.shape[0]
        n_actions = self.n_actions

        P_cf = self.cf_sample_prob(trajectories, n_actions, n_steps)

        return P_cf


    def run_sample(self, inp, trajectory, P_cf):
        P_cf_save = {}

        for i in inp:
            self.cf_sample_prob_parallel(trajectory, i[0], i[1], P_cf_save)

        for i in inp:
            P_cf.update(P_cf_save[i])


    def run_gumbel_sampling_parallel_helper(self, trajectories):
        n_steps = trajectories.shape[0]
        n_actions = self.n_actions
        
        inp = [(a, time_idx) for time_idx in range(n_steps) for a in range(n_actions)]

        # Run with n threads.
        def split(a, n):
            k, m = divmod(len(a), n)
            return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
        
        split_work = split(inp, 32)
        processes = []

        with Manager() as manager:
            P_cf = manager.dict()
            
            for chunk in split_work:
                process = Process(target=self.run_sample, args=(chunk, trajectories, P_cf))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            return P_cf.copy()
        

    def run_gumbel_sampling_parallel(self, observed_path):
        n_steps = observed_path.shape[0]
        n_actions = self.n_actions
        n_states = self.n_states

        cf_mdp = np.zeros(shape=(n_steps, n_states, n_actions, n_states))

        P_cf = self.run_gumbel_sampling_parallel_helper(observed_path)

        for t in range(n_steps):
            for s in range(n_states):
                for a in range(n_actions):
                    cf_mdp[t, s, a] = P_cf[a, t][s]

        return cf_mdp
    

class GumbelMaxCFBoundCalculatorIMDP(object):
    def __init__(self, mdp, imdp, n_timesteps, n_states, n_actions):
        self.mdp = mdp
        self.imdp = imdp
        self.n_timesteps = n_timesteps
        self.n_states = n_states
        self.n_actions = n_actions


    def truncated_gumbel(self, logit, truncation):
        assert not np.isneginf(logit)

        gumbel = np.random.gumbel(size=(truncation.shape[0])) + logit
        trunc_g = -np.log(np.exp(-gumbel) + np.exp(-truncation))
        return trunc_g


    def topdown(self, obs_logits, obs_state, nsamp=1):
        poss_next_states = obs_logits.shape[0]
        gumbels = np.zeros((nsamp, poss_next_states))

        # Sample top gumbels.
        topgumbel = np.random.gumbel(size=(nsamp))

        for next_state in range(poss_next_states):
            # This is the observed next state.
            if (next_state == obs_state) and not(np.isneginf(obs_logits[next_state])):
                gumbels[:, obs_state] = topgumbel - obs_logits[next_state]

            # These were the other feasible options (p > 0).
            elif not(np.isneginf(obs_logits[next_state])):
                gumbels[:, next_state] = self.truncated_gumbel(obs_logits[next_state], topgumbel) - obs_logits[next_state]

            # These had zero probability to start with, so are unconstrained.
            else:
                gumbels[:, next_state] = np.random.gumbel(size=nsamp)

        return gumbels
    

    def normalise_gumbel_icfmdp(self, interval_CF_MDP):
        def trunc_to_decimal_places(value, decimal_places):
            factor = 10 ** decimal_places
            return np.trunc(value * factor) / factor
        
        def ceil_to_decimal_places(value, decimal_places):
            factor = 10 ** decimal_places
            return np.ceil(value * factor) / factor
        
        interval_CF_MDP[:, :, :, :, 0] = trunc_to_decimal_places(interval_CF_MDP[:, :, :, :, 0], 12)
        interval_CF_MDP[:, :, :, :, 1] = ceil_to_decimal_places(interval_CF_MDP[:, :, :, :, 1], 12)

        # Normalising
        for t in range(self.n_timesteps):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    ub_threshold = 1.0000000000000000
                    lb_threshold = 1.0000000000000000

                    i = 0

                    while sum(interval_CF_MDP[t, s, a, :, 0]) > lb_threshold:
                        #print(f"iter i {i} s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")
            
                        interval_CF_MDP[t, s, a, :, 0] = interval_CF_MDP[t, s, a, :, 0] / sum(interval_CF_MDP[t, s, a, :, 0])
                        
                        #print(f"iter i {i} s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")

                        # if sum(interval_CF_MDP[t, s, a, :, 0]) > lb_threshold:
                            #print(f"iter i {i} s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")
                        i += 1
                    
                    i=0
                    while sum(interval_CF_MDP[t, s, a, :, 1]) < ub_threshold:
                        #print(f"iter {i} s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")
                
                        interval_CF_MDP[t, s, a, :, 1] = interval_CF_MDP[t, s, a, :, 1] / sum(interval_CF_MDP[t, s, a, :, 1])
                        
                        #print(f"iter {i} s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")

                        # if sum(interval_CF_MDP[t, s, a, :, 1]) < ub_threshold:
                            # print(f"iter {i} s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")
                        i += 1

                    # if sum(interval_CF_MDP[t, s, a, :, 1]) < ub_threshold:
                        #print(f"s={s} a={a} ub={interval_CF_MDP[t, s, a, :, 1]} sum={sum(interval_CF_MDP[t, s, a, :, 1])}")

                    # if sum(interval_CF_MDP[t, s, a, :, 0]) > lb_threshold:
                        #print(f"s={s} a={a} lb={interval_CF_MDP[t, s, a, :, 0]} sum={sum(interval_CF_MDP[t, s, a, :, 0])}")

                    assert(sum(interval_CF_MDP[t, s, a, :, 1]) >= ub_threshold)
                    assert(sum(interval_CF_MDP[t, s, a, :, 0]) <= lb_threshold)

        epsilon = 1e-16
        nonzero_mask = interval_CF_MDP[:, :, :, :, 0] != 0.0
        # This adjustment ensure the probs are valid (i.e., that all UBs are > LBs)
        interval_CF_MDP[nonzero_mask, 0] -= epsilon
        nonzero_mask = interval_CF_MDP[:, :, :, :, 1] != 0.0
        interval_CF_MDP[nonzero_mask, 1] += epsilon

        return interval_CF_MDP


    def cf_posterior(self, obs_prob, intrv_prob, next_state, n_mc):
        obs_logits = np.log(obs_prob)
        intrv_logits = np.log(intrv_prob)
        gumbels = self.topdown(obs_logits, next_state, n_mc)

        posterior = intrv_logits + gumbels
        intrv_posterior = np.argmax(posterior, axis=1)
        posterior_prob = np.zeros(np.size(intrv_prob, 0))
        
        for i in range(np.size(intrv_prob, 0)):
            posterior_prob[i] = np.sum(intrv_posterior == i) / n_mc

        return posterior_prob, intrv_posterior
    

    def generate_simplex_grid(self, low, high, n_samples):
        """
        Generate a discretized grid over the probability simplex that lies within given [low, high] bounds.
        - low, high: 1D arrays of shape (d,) with 0 <= low[i] <= high[i] <= 1
        - n_points: granularity of discretization (higher = finer grid)
        """
        dim = len(low)
        step = 1.0 / n_samples

        # Generate all combinations of integer tuples (i_1, ..., i_d) such that sum(i_k) = n_points
        grids = []
        for counts in itertools.product(range(n_samples + 1), repeat=dim):
            if sum(counts) != n_samples:
                continue
            vec = np.array(counts, dtype=np.float64) / n_samples
            if np.all(vec >= low) and np.all(vec <= high):
                grids.append(vec)

        return np.array(grids)


    def sample_from_interval_simplex(self, low, high, n_samples):
        dim = len(low)
        samples = []
        
        for _ in range(n_samples):
            # Dirichlet sample
            p = np.random.dirichlet(np.ones(dim))
            scaled = low + p * (high - low)
            scaled /= np.sum(scaled)  # re-normalize
            if np.all(scaled >= low) and np.all(scaled <= high) and np.isclose(np.sum(scaled), 1.0):
                samples.append(scaled)
        
        return np.array(samples)


    def _cf_sample_prob_single_tight(self, args):
        t, s, a, s_prime, observed_path, n_mc = args

        obs_state = int(observed_path[t][0])
        obs_action = int(observed_path[t][2])
        obs_next_state = int(observed_path[t][1])

        obs_candidates = self.sample_from_interval_simplex(self.imdp[obs_state, obs_action, :, 0], self.imdp[obs_state, obs_action, :, 1], n_samples=10)
        cf_candidates = self.sample_from_interval_simplex(self.imdp[s, a, :, 0], self.imdp[s, a, :, 1], n_samples=10)

        # #print(obs_candidates.shape)
        # #print(cf_candidates.shape)

        # if obs_candidates.shape[0] == 0 or cf_candidates.shape[0] == 0:
            #print(self.imdp[obs_state, obs_action, :, :])
            #print(self.imdp[s, a, :, :])
            #print(obs_candidates)
            #print(cf_candidates)
            #print(f"{self.imdp[s, a, :, 0]}, {self.imdp[s, a, :, 1]} cf")

        assert(obs_candidates.shape[0] > 0)
        assert(cf_candidates.shape[0] > 0)

        cf_probs = []
        for obs_intrv in obs_candidates:
            for cf_intrv in cf_candidates:
                prob, _ = self.cf_posterior(obs_intrv, cf_intrv, obs_next_state, n_mc)
                cf_probs.append(prob)

        cf_probs = np.array(cf_probs)
        # #print(cf_probs.shape)
        assert(cf_probs.shape[0] > 0)

        cf_prob_lower = np.min(cf_probs[:, s_prime])
        cf_prob_upper = np.max(cf_probs[:, s_prime])

        return (t, s, a, s_prime, cf_prob_lower, cf_prob_upper)


    def _cf_sample_prob_single(self, args):
        t, s, a, s_prime, observed_path, n_mc = args

        obs_state = int(observed_path[t][0])
        obs_action = int(observed_path[t][2])
        obs_next_state = int(observed_path[t][1])

        # These bounds are valid but not tight.
        # Upper bound calculation
        obs_intrv = self.imdp[obs_state, obs_action, :, 1]
        obs_intrv[s_prime] = self.imdp[obs_state, obs_action, s_prime, 0]
        cf_intrv = self.imdp[s, a, :, 0].copy()
        cf_intrv[s_prime] = self.imdp[s, a, s_prime, 1]

        cf_prob_upper, _ = self.cf_posterior(obs_intrv, cf_intrv, obs_next_state, n_mc)

        # Lower bound calculation
        obs_intrv = self.imdp[obs_state, obs_action, :, 0]
        obs_intrv[s_prime] = self.imdp[obs_state, obs_action, s_prime, 1]
        cf_intrv = self.imdp[s, a, :, 1]
        cf_intrv[s_prime] = self.imdp[s, a, s_prime, 0]

        cf_prob_lower, _ = self.cf_posterior(obs_intrv, cf_intrv, obs_next_state, n_mc)

        return (t, s, a, s_prime, cf_prob_lower[s_prime], cf_prob_upper[s_prime])
    

    def cf_sample_prob(self, observed_path): 
        n_steps = observed_path.shape[0]
        n_mc = 10000

        cf_imdp = np.zeros(shape=(n_steps, self.n_states, self.n_actions, self.n_states, 2))

        tasks = []
        for t in range(n_steps):
            for (s, a, s_prime) in self.mdp.get_possible_transitions():
                tasks.append((t, s, a, s_prime, observed_path, n_mc))

        with mp.Pool(processes=32) as pool:
            results = pool.map(self._cf_sample_prob_single, tasks)

        for (t, s, a, s_prime, lb, ub) in results:
            lb = max(0.0, lb)
            lb = min(1.0, lb)
            ub = max(0.0, ub)
            ub = min(1.0, ub)

            if lb > ub:
                #print(lb)
                #print(ub)
                assert(math.isclose(lb, ub, rel_tol=0.02, abs_tol=0.02))
                # If they're close, pick one - this is due to sampling error.
                ub = lb
                
            cf_imdp[t, s, a, s_prime, 0] = lb
            cf_imdp[t, s, a, s_prime, 1] = ub

        cf_imdp = self.normalise_gumbel_icfmdp(cf_imdp)

        return cf_imdp
    

    def cf_sample_prob_parallel(self, trajectory, a, time_idx, P_cf_save, n_cf_samps=1): 
        n_steps = trajectory.shape[0]
        n_mc = 10000
        P_cf = {}

        for _ in range(n_cf_samps):
            obs_transition = trajectory[time_idx]
            obs_state = int(obs_transition[0])
            obs_next_state = int(obs_transition[1])
            obs_action = int(obs_transition[2])

            P_cf[a, time_idx] = np.zeros((int(self.n_states),int(self.n_states), 2))
            
            for s in range(self.n_states):
                for s_prime in range(self.n_states):
                    # These bounds are valid but not tight.

                    # Upper bound calculation
                    obs_intrv = self.imdp[obs_state, obs_action, :, 1]
                    obs_intrv[s_prime] = self.imdp[obs_state, obs_action, s_prime, 0]
                    cf_intrv = self.imdp[s, a, :, 0]
                    cf_intrv[s_prime] = self.imdp[s, a, s_prime, 1]

                    cf_prob_upper, _ = self.cf_posterior(obs_intrv, cf_intrv, obs_next_state, n_mc)

                    # Lower bound calculation
                    obs_intrv = self.imdp[obs_state, obs_action, :, 0]
                    obs_intrv[s_prime] = self.imdp[obs_state, obs_action, s_prime, 1]
                    cf_intrv = self.imdp[s, a, :, 1]
                    cf_intrv[s_prime] = self.imdp[s, a, s_prime, 0]

                    cf_prob_lower, _ = self.cf_posterior(obs_intrv, cf_intrv, obs_next_state, n_mc)

                    P_cf[a,time_idx][s, s_prime, 0] = cf_prob_lower[s_prime]
                    P_cf[a,time_idx][s, s_prime, 1] = cf_prob_upper[s_prime]

        cf_imdp = np.zeros(shape=(n_steps, self.n_states, self.n_actions, self.n_states, 2))

        P_cf_save[(a,time_idx)] = P_cf

        return cf_imdp
    

    def cf_sample_prob_tight(self, observed_path): 
        n_steps = observed_path.shape[0]
        n_mc = 10000

        cf_imdp = np.zeros(shape=(n_steps, self.n_states, self.n_actions, self.n_states, 2))

        tasks = []
        
        for t in range(n_steps):
            for (s, a, s_prime) in self.mdp.get_possible_transitions():
                tasks.append((t, s, a, s_prime, observed_path, n_mc))

        with mp.Pool(processes=32) as pool:
            results = pool.map(self._cf_sample_prob_single_tight, tasks)

        for (t, s, a, s_prime, lb, ub) in results:
            cf_imdp[t, s, a, s_prime, 0] = lb
            cf_imdp[t, s, a, s_prime, 1] = ub

        return cf_imdp


    def run_gumbel_sampling(self, observed_path):
        return self.cf_sample_prob(observed_path)
    

    def run_gumbel_sampling_tight(self, observed_path):
        return self.cf_sample_prob_tight(observed_path)
    

    def run_sample(self, inp, trajectory, P_cf):
        P_cf_save = {}

        for i in inp:
            self.cf_sample_prob_parallel(trajectory, i[0], i[1], P_cf_save)

        for i in inp:
            P_cf.update(P_cf_save[i])
