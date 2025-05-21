import numpy as np
from decimal import Decimal
from multiprocessing import Process, Manager

class CFBoundCalculator:
    def __init__(self, observed_state, observed_action, observed_next_state, transition_matrix):
        self.observed_state = observed_state
        self.observed_action = observed_action
        self.observed_next_state = observed_next_state
        self.transition_matrix = transition_matrix

    def _get_support(self, s, a):
        return set(np.nonzero(self.transition_matrix[s, a, :])[0])


    def _check_counterfactual_stability(self, s, a, s_prime):
        if self.transition_matrix[self.observed_state, self.observed_action, s_prime] > 0.0 and not s_prime == self.observed_next_state:
            if (self.transition_matrix[s, a, self.observed_next_state] / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]) >= (self.transition_matrix[s, a, s_prime] / self.transition_matrix[self.observed_state, self.observed_action, s_prime]):
                return False
            
        return True


    def _lower_bound_helper(self, s, a, s_prime):
        support_of_cf = self._get_support(s, a)
        support_of_cf.discard(s_prime)

        upper_bounds = sum([self.transition_matrix[s, a, s_prime_prime] - (self.upper_bound(s, a, s_prime_prime) * self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]) for s_prime_prime in support_of_cf])

        u_2_remaining = 1 - self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state] - upper_bounds

        return (self.transition_matrix[s, a, s_prime] - u_2_remaining) / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]


    def lower_bound(self, s, a, s_prime):
        if s == self.observed_state and a == self.observed_action:
            # (s_t, a_t) -> s'
            if s_prime == self.observed_next_state:
                return 1.0
            else:
                return 0.0
            
        support_of_observed = self._get_support(self.observed_state, self.observed_action)
        support_of_cf = self._get_support(s, a)
        overlapping_support = support_of_observed.intersection(support_of_cf)

        if len(overlapping_support) == 0:
            # (s, a) has disjoint support with (s_t, a_t).
            if self.transition_matrix[s, a, s_prime] > 1 - self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]:
                return (self.transition_matrix[s, a, s_prime] - (1 - self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state])) / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]
            else:
                return 0.0
            
        # (s, a) has overlapping support with (s_t, a_t).
        if s_prime != self.observed_next_state:
            if not self._check_counterfactual_stability(s, a, s_prime):
                # CF prob must be 0 to satisfy counterfactual stability.
                return 0.0
            
            # Transition vacuously satisfies counterfactual stability.
            return max(0.0, self._lower_bound_helper(s, a, s_prime))

        else:
            # (s, a) -> s_t+1
            return max(self.transition_matrix[s, a, s_prime], self._lower_bound_helper(s, a, s_prime))


    def upper_bound(self, s, a, s_prime):
        if s == self.observed_state and a == self.observed_action:
            if s_prime == self.observed_next_state:
                return 1.0
            else:
                return 0.0
            
        support_of_observed = self._get_support(self.observed_state, self.observed_action)
        support_of_cf = self._get_support(s, a)
        overlapping_support = support_of_observed.intersection(support_of_cf)

        if len(overlapping_support) > 0:
            if s_prime != self.observed_next_state:
                if self._check_counterfactual_stability(s, a, s_prime):
                    pass
                    if s_prime in support_of_observed:
                        return min(1 - self.transition_matrix[s, a, self.observed_next_state], self.transition_matrix[s, a, s_prime])
                    else:
                        return min(1 - self.transition_matrix[s, a, self.observed_next_state], self.transition_matrix[s, a, s_prime] / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state])
                else:
                    return 0
                
            else:
                if self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state] <= self.transition_matrix[s, a, s_prime]:
                    return 1.0
                else:
                    return self.transition_matrix[s, a, s_prime] / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]
            
        # (s, a) has disjoint support from (s_t, a_t).
        return min(self.transition_matrix[s, a, s_prime], self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]) / self.transition_matrix[self.observed_state, self.observed_action, self.observed_next_state]


    def calculate_all_bounds(self):
        n_states = self.transition_matrix.shape[0]
        n_actions = self.transition_matrix.shape[1]

        interval_cf_transition_matrix = np.zeros(shape=(n_states, n_actions, n_states, 2))

        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    lb = Decimal(self.lower_bound(s, a, s_prime))
                    ub = Decimal(self.upper_bound(s, a, s_prime))
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
        n_states = self.transition_matrix.shape[0]
        n_actions = self.transition_matrix.shape[1]

        interval_cf_transition_matrix = np.zeros(shape=(n_states, n_actions, n_states, 2))

        transitions = [(s, a, s_prime) for s in range(n_states) for a in range(n_actions) for s_prime in range(n_states)]

        upper_bounds, lower_bounds = self.run_parallel(transitions)
        
        for s in range(n_states):
            for a in range(n_actions):
                for s_prime in range(n_states):
                    interval_cf_transition_matrix[s, a, s_prime] = [upper_bounds[(s, a, s_prime)], lower_bounds[(s, a, s_prime)]]
                
        return interval_cf_transition_matrix
    

class MultiStepCFBoundCalculator:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix
    

    def calculate_bounds(self, trajectory):
        n_timesteps = len(trajectory)
        n_states = self.transition_matrix.shape[0]
        n_actions = self.transition_matrix.shape[1]

        interval_cf_transition_matrix = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states, 2))

        for t in range(n_timesteps):
            print(f"Calculating bounds at time t={t}")
            bound_calculator = CFBoundCalculator(trajectory[t][0], trajectory[t][2], trajectory[t][1], self.transition_matrix)
            interval_cf_transition_matrix[t] = bound_calculator.calculate_all_bounds()

        return interval_cf_transition_matrix


class ParallelMultiStepCFBoundCalculator:
    def __init__(self, transition_matrix):
        self.transition_matrix = transition_matrix

    
    def run_parallel(self, t, trajectory, bounds):
        bound_calculator = CFBoundCalculator(trajectory[t][0], trajectory[t][2], trajectory[t][1], self.transition_matrix)
        
        bounds[t] = bound_calculator.calculate_all_bounds()
    

    def calculate_bounds(self, trajectory):
        n_timesteps = len(trajectory)
        n_states = self.transition_matrix.shape[0]
        n_actions = self.transition_matrix.shape[1]

        interval_cf_transition_matrix = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states, 2))

        processes = []

        def foo():
            with Manager() as manager:
                bounds = manager.dict()

                for t in range(n_timesteps):
                    process = Process(target=self.run_parallel, args=(t, trajectory, bounds))
                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()

                return bounds.copy()
            
        bounds = foo()

        for t in range(n_timesteps):
            interval_cf_transition_matrix[t] = bounds[t]
                
        return interval_cf_transition_matrix