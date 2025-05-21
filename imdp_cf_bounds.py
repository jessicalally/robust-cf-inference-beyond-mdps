import cvxpy as cp
import math
import numpy as np
from decimal import Decimal
from multiprocessing import Process, Manager

# Provides methods for calculating the CF probabilities of a data-driven IMDP.

class CFBoundCalculator:
    def __init__(self, observed_state, observed_action, observed_next_state, imdp_transition_matrix):
        self.observed_state = observed_state
        self.observed_action = observed_action
        self.observed_next_state = observed_next_state
        self.imdp_transition_matrix = imdp_transition_matrix

    def _get_support(self, s, a):
        # For an IMDP, the support of a distribution is every possible state that can be transitioned too.
        # Therefore, we look at the upper bound transition probabilities.
        return set(np.nonzero(self.imdp_transition_matrix[s, a, :, 1])[0])


    # This checks whether the counterfactual stability assumption trivially holds, or it always limits the CF prob of the transition to 0.
    def _always_counterfactual_stability(self, s, a, s_prime):
        if s_prime != self.observed_next_state and self.imdp_transition_matrix[self.observed_state, self.observed_action, s_prime, 0] > 0.0:
            if (self.imdp_transition_matrix[s, a, self.observed_next_state, 0] / self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 1]) > (self.imdp_transition_matrix[s, a, s_prime, 1] / self.imdp_transition_matrix[self.observed_state, self.observed_action, s_prime, 0]):
                return True
            
        return False
    
    # This checks whether the counterfactual stability assumption could limit the CF prob of the transition to 0.
    def _sometimes_counterfactual_stability(self, s, a, s_prime):
        if s_prime != self.observed_next_state and self.imdp_transition_matrix[self.observed_state, self.observed_action, s_prime, 1] > 0.0:
            if (self.imdp_transition_matrix[s, a, self.observed_next_state, 1] / self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]) > (self.imdp_transition_matrix[s, a, s_prime, 0] / self.imdp_transition_matrix[self.observed_state, self.observed_action, s_prime, 1]):
                return True
            
        return False
    

    def _optimise_sum_max_ubs_of_other_transitions(self, s, a, s_prime):
        # TODO: can we instead rewrite this as extra constraints? e.g., the cs constraint
        def _exact_upper_bound(P_obs, P_cf, s_prime_prime):
            if s == self.observed_state and a == self.observed_action:
                if s_prime_prime == self.observed_next_state:
                    return 1.0
                else:
                    return 0.0
                
            # TODO: we can't evaluate the support, because it depends on the cvxpy variables. In most cases we know that the supports overlap. But, depending on the sampled probs from the IMDP, we may sample something that has disjoint support.
            # TODO: to solve this, we may be able to add more conditions into the problem, and only run the optimisation if we really have to.

            # support_of_observed = set(np.nonzero(P_obs)[0])
            # support_of_cf = set(np.nonzero(P_cf)[0])
            # overlapping_support = support_of_observed.intersection(support_of_cf)

            # if len(overlapping_support) > 0:
            if s_prime_prime != self.observed_next_state:
                if P_obs[s_prime_prime] > 0 and not s_prime_prime == self.observed_next_state:
                    if (P_cf[self.observed_next_state] / P_obs[self.observed_next_state]) > (P_cf[s_prime_prime] / P_obs[s_prime_prime]):
                        return 0.0
                    
                if P_obs[s_prime_prime] > 0:
                    return min(1 - P_cf[self.observed_next_state], P_cf[s_prime_prime])
                else:
                    return min(1 - P_cf[self.observed_next_state], P_cf[s_prime_prime] / P_obs[self.observed_next_state])
                
            else:
                if P_obs[self.observed_next_state] <= P_cf[s_prime_prime]:
                    return 1.0
                else:
                    return P_cf[s_prime_prime] / P_obs[self.observed_next_state]
                
            # (s, a) has disjoint support from (s_t, a_t).
            # return min(P_cf[s_prime_prime], P_obs[self.observed_next_state]) / P_obs[self.observed_next_state]

        n_states = self.imdp_transition_matrix.shape[0]

        P_obs = cp.Variable(n_states)
        P_cf = cp.Variable(n_states)

        constraints = [
            cp.sum(P_obs) == 1,
            cp.sum(P_cf) == 1,
            P_obs >= self.imdp_transition_matrix[self.observed_state, self.observed_action, :, 0],
            P_obs <= self.imdp_transition_matrix[self.observed_state, self.observed_action, :, 1],
            P_cf >= self.imdp_transition_matrix[s, a, :, 0],
            P_cf <= self.imdp_transition_matrix[s, a, :, 1]
        ]

        objective_expr = cp.sum([_exact_upper_bound(P_obs, P_cf, i) for i in range(n_states) if i != s_prime])
        objective = cp.Maximize(objective_expr)

        prob = cp.Problem(objective, constraints)
        prob.solve()

        print("Optimal value:", prob.value)
        print("Optimal P(.|s_t, a_t):", P_obs.value)
        print("Optimal P(.|s, a):", P_cf.value)


    def lower_bound(self, s, a, s_prime):
        if s == self.observed_state and a == self.observed_action:
            if s_prime == self.observed_next_state:
                return 1.0
            else:
                return 0.0
                        
        support_of_observed = self._get_support(self.observed_state, self.observed_action)
        support_of_cf = self._get_support(s, a)
        overlapping_support = support_of_observed.intersection(support_of_cf)

        if len(overlapping_support) == 0:
            # The supports are fully disjoint.
            if self.imdp_transition_matrix[s, a, s_prime, 0] > 1 - self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]:
                return (self.imdp_transition_matrix[s, a, s_prime, 0] - (1 - self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0])) / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]
            else:
                return 0.0 
 
        # Otherwise, the supports overlap.
        
        # This implementation is non-tight, this sum produces a non-tight LB. But, we could replace this calculation with an optimisation problem to find the exact LB.
        # TODO: need to test the optimisation problem, to see (a) if it is possible and (b) how much slower it is.
        def _sum_max_ubs_of_other_transitions():
            n_states = self.imdp_transition_matrix.shape[0]
            sum_ubs = 0.0

            for s_prime_prime in range(n_states):
                if s_prime != s_prime_prime:
                    sum_ubs += self.upper_bound(s, a, s_prime_prime)

            return sum_ubs

        # Case 1: s' = s_{t+1}
        if s_prime == self.observed_next_state:
            return max(self.imdp_transition_matrix[s, a, s_prime, 0], 1 - _sum_max_ubs_of_other_transitions())#self._optimise_sum_max_ubs_of_other_transitions(s, a, s_prime))
        
        # Case 2: counterfactual stability could limit the CF prob to 0.
        elif self._sometimes_counterfactual_stability(s, a, s_prime):
            return 0
        
        # Case 3: all other cases.
        return max(0, 1 - _sum_max_ubs_of_other_transitions()) #self._optimise_sum_max_ubs_of_other_transitions(s, a, s_prime))
            

    def upper_bound(self, s, a, s_prime):
        if s == self.observed_state and a == self.observed_action:
            if s_prime == self.observed_next_state:
                return 1.0
            else:
                return 0.0
                        
        support_of_observed = self._get_support(self.observed_state, self.observed_action)
        support_of_cf = self._get_support(s, a)
        overlapping_support = support_of_observed.intersection(support_of_cf)

        if len(overlapping_support) == 0:
            # The supports are fully disjoint.
            return min(self.imdp_transition_matrix[s, a, s_prime, 1], self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]) / self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]
 
        # Otherwise, the supports overlap.

        # Case 1: s' = s_{t+1}.
        if s_prime == self.observed_next_state:
            return min(self.imdp_transition_matrix[s, a, s_prime, 1], self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]) / self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]
        
        # Case 2: counterfactual stability limits the CF prob to 0.
        elif self._always_counterfactual_stability(s, a, s_prime):
            return 0
        
        # Case 3: monotonicity assumption always affects CF probs.
        elif self.imdp_transition_matrix[self.observed_state, self.observed_action, s_prime, 0] > 0.0:
            return min(self.imdp_transition_matrix[s, a, s_prime, 1], 1 - self.imdp_transition_matrix[s, a, self.observed_next_state, 0])
        
        # Case 4: monotonicity assumption never affect CF probs.
        elif self.imdp_transition_matrix[self.observed_state, self.observed_action, s_prime, 1] == 0.0:
            return min(self.imdp_transition_matrix[s, a, s_prime, 1] / self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0], 1 - self.imdp_transition_matrix[s, a, self.observed_next_state, 0])

        # Case 5: monotonicity assumption sometimes affects CF probs.
        sum_of_ubs = np.sum(self.imdp_transition_matrix[self.observed_state, self.observed_action, :, 1]) - self.imdp_transition_matrix[self.observed_state, self.observed_action, s_prime, 1]
        
        if sum_of_ubs >= 1.0:
            return min(self.imdp_transition_matrix[s, a, s_prime, 1] / max(1 - sum_of_ubs - self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 1], self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]), 1 - self.imdp_transition_matrix[s, a, self.observed_next_state, 0])

        # Case 6: all other cases.
        return min(self.imdp_transition_matrix[s, a, s_prime, 1], 1 - self.imdp_transition_matrix[s, a, self.observed_next_state, 0])
        

    def calculate_all_bounds(self):
        n_states = self.imdp_transition_matrix.shape[0]
        n_actions = self.imdp_transition_matrix.shape[1]

        interval_cf_transition_matrix = np.zeros(shape=(n_states, n_actions, n_states, 2))

        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    lb = Decimal(self.lower_bound(s, a, s_prime))
                    assert(not math.isnan(lb))
                    ub = Decimal(self.upper_bound(s, a, s_prime))
                    assert(not math.isnan(ub))
                    interval_cf_transition_matrix[s, a, s_prime] = [lb, ub]
                
        return interval_cf_transition_matrix
    

    # Run with n threads
    def split_work(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


    def calculate_upper_bounds(self, chunk, upper_bounds):
        for (s, a, s_prime) in chunk:
            ub = Decimal(self.upper_bound(s, a, s_prime))
            upper_bounds[(s, a, s_prime)] = ub


    def calculate_lower_bounds(self, chunk, lower_bounds):
        for (s, a, s_prime) in chunk:
            lb = Decimal(self.lower_bound(s, a, s_prime))
            lower_bounds[(s, a, s_prime)] = lb


    def run_parallel(self, transitions):
        processes = []

        with Manager() as manager:
            upper_bounds = manager.dict()

            # Calculate all of the upper bounds first, then all the lower bounds, since we need the upper bounds
            # to calculate the lower bounds.
            for chunk in self.split_work(transitions, 32):
                process = Process(target=self.calculate_upper_bounds, args=(chunk, upper_bounds))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            lower_bounds = manager.dict()

            for chunk in self.split_work(transitions, 32):
                process = Process(target=self.calculate_lower_bounds, args=(chunk, lower_bounds))
                processes.append(process)
                process.start()

            for process in processes:
                process.join()

            return upper_bounds.copy(), lower_bounds.copy()


    def parallel_calculate_all_bounds(self):
        n_states = self.imdp_transition_matrix.shape[0]
        n_actions = self.imdp_transition_matrix.shape[1]

        interval_cf_transition_matrix = np.zeros(shape=(n_states, n_actions, n_states, 2))

        transitions = [(s, a, s_prime) for s in range(n_states) for a in range(n_actions) for s_prime in range(n_states)]

        upper_bounds, lower_bounds = self.run_parallel(transitions)
        
        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    interval_cf_transition_matrix[s, a, s_prime] = [upper_bounds[(s, a, s_prime)], lower_bounds[(s, a, s_prime)]]
                
        return interval_cf_transition_matrix
    

class MultiStepCFBoundCalculator:
    def __init__(self, imdp_transition_matrix):
        self.imdp_transition_matrix = imdp_transition_matrix


    def calculate_bounds(self, trajectory):
        n_timesteps = len(trajectory)
        n_states = self.imdp_transition_matrix.shape[0]
        n_actions = self.imdp_transition_matrix.shape[1]
        assert(self.imdp_transition_matrix.shape[2] == n_states)
        assert(self.imdp_transition_matrix.shape[3] == 2)

        interval_cf_transition_matrix = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states, 2))

        for t in range(n_timesteps):
            print(f"Calculating bounds at time t={t}")
            bound_calculator = CFBoundCalculator(trajectory[t][0], trajectory[t][2], trajectory[t][1], self.imdp_transition_matrix)
            interval_cf_transition_matrix[t] = bound_calculator.calculate_all_bounds()

        return interval_cf_transition_matrix


class ParallelMultiStepCFBoundCalculator:
    def __init__(self, imdp_transition_matrix):
        self.imdp_transition_matrix = imdp_transition_matrix

    
    def run_parallel(self, t, trajectory, bounds):
        bound_calculator = CFBoundCalculator(trajectory[t][0], trajectory[t][2], trajectory[t][1], self.imdp_transition_matrix)
        
        bounds[t] = bound_calculator.calculate_all_bounds()
    

    def calculate_bounds(self, trajectory):
        n_timesteps = len(trajectory)
        n_states = self.imdp_transition_matrix.shape[0]
        n_actions = self.imdp_transition_matrix.shape[1]
        assert(self.imdp_transition_matrix.shape[2] == n_states)
        assert(self.imdp_transition_matrix.shape[3] == 2)

        interval_cf_transition_matrix = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states, 2))

        processes = []

        def _start_processes():
            with Manager() as manager:
                bounds = manager.dict()

                for t in range(n_timesteps):
                    process = Process(target=self.run_parallel, args=(t, trajectory, bounds))
                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()

                return bounds.copy()
            
        bounds = _start_processes()

        for t in range(n_timesteps):
            interval_cf_transition_matrix[t] = bounds[t]
                
        return interval_cf_transition_matrix