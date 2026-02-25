import os
import gurobipy as gp
from gurobipy import GRB
import math
import numpy as np
from decimal import Decimal
from joblib import Parallel, delayed
import multiprocessing as mp
from multiprocessing import Process, Manager

# Provides methods for calculating the CF probabilities of a data-driven IMDP.

_worker_env = None

def _init_worker():
    global _worker_env
    _worker_env = gp.Env(empty=True)
    _worker_env.setParam('OutputFlag', 0)
    _worker_env.start()

class CFBoundCalculator:
    def __init__(self, mdp, observed_state, observed_action, observed_next_state, imdp_transition_matrix):
        self.mdp = mdp
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
            assert(self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0] > 0)
            assert(self.imdp_transition_matrix[self.observed_state, self.observed_action, s_prime, 1] > 0)
            
            if (self.imdp_transition_matrix[s, a, self.observed_next_state, 1] / self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]) > (self.imdp_transition_matrix[s, a, s_prime, 0] / self.imdp_transition_matrix[self.observed_state, self.observed_action, s_prime, 1]):
                return True
            
        return False
    

    def _optimise_sum_max_ubs_of_other_transitions(self, model, s, a, s_prime):
        # Checks to ensure optimisation is only run if necessary.

        if not s_prime == self.observed_next_state:
            # UB of transition to observed next state is at least min(P(s_{t+1} \mid s, a), P(s_{t+1} \mid s_t, a_t))/P(s_{t+1} \mid s_t, a_t)
            if min(self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0], self.imdp_transition_matrix[s, a, self.observed_next_state, 1]) / self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0] >= 1.0:
                # Upper bounds are guaranteed to sum to at least 1.0.
                return 1.0

        n_states = self.imdp_transition_matrix.shape[0]

        P_obs = {} # Interventional distribution of observed state-action pair.
        P_cf = {} # Interventional distribution of CF state-action pair.
        upper_cf_bounds = {} # Stores the optimised upper cf bound for each state.

        M = 1e3 # Very big value.
        epsilon = 1e-6  # Very small value.

        # Define transition variables.
        for i in range(n_states):
            P_obs[i] = model.addVar(lb=0, ub=1, name=f"P_obs_{i}")
            P_cf[i] = model.addVar(lb=0, ub=1, name=f"P_cf_{i}")

        # Probabilities must sum to 1.
        model.addConstr(gp.quicksum(P_obs[i] for i in range(n_states)) == 1)
        model.addConstr(gp.quicksum(P_cf[i] for i in range(n_states)) == 1)

        # Apply lower/upper bounds from IMDP to interventional distributions.
        for i in range(n_states):
            P_obs[i].LB = float(self.imdp_transition_matrix[self.observed_state, self.observed_action, i, 0])
            P_obs[i].UB = float(self.imdp_transition_matrix[self.observed_state, self.observed_action, i, 1])
            P_cf[i].LB = float(self.imdp_transition_matrix[s, a, i, 0])
            P_cf[i].UB = float(self.imdp_transition_matrix[s, a, i, 1])

        # Indicator variables that determine whether the supports are disjoint or overlap during optimisation.
        support_obs = {} # Indicates whether states are in the support of the observed state-action pair.
        support_cf = {} # Indicates whether states are in the support of the CF state-action pair.
        support_overlap = {} # Indicates whether states are in both supports.

        # Detects overlapping support.
        for i in range(n_states):
            support_obs[i] = model.addVar(vtype=GRB.BINARY, name=f"support_obs_{i}")
            support_cf[i] = model.addVar(vtype=GRB.BINARY, name=f"support_cf_{i}")
            support_overlap[i] = model.addVar(vtype=GRB.BINARY, name=f"support_overlap_{i}")

            # support_obs[i] = 1 <-> P_obs[i] >= epsilon.
            model.addConstr(P_obs[i] >= epsilon * support_obs[i])
            model.addConstr(P_obs[i] <= support_obs[i])

            # support_cf[i] = 1 <-> P_cf[i] >= epsilon.
            model.addConstr(P_cf[i] >= epsilon * support_cf[i])
            model.addConstr(P_cf[i] <= support_cf[i])

            # support_ovelap[i] = 1 <-> support_obs[i] = 1 AND support_cf[i] = 1.
            model.addConstr(support_overlap[i] <= support_obs[i])
            model.addConstr(support_overlap[i] <= support_cf[i])
            model.addConstr(support_overlap[i] >= support_obs[i] + support_cf[i] - 1)

        # total_ovelap evaluates the number of states in the support overlap.
        total_overlap = model.addVar(name="total_overlap")
        model.addConstr(total_overlap == gp.quicksum(support_overlap[i] for i in range(n_states)))

        # is_disjoint = 1 <-> supports are disjoint.
        is_disjoint = model.addVar(vtype=GRB.BINARY, name="is_disjoint")
        model.addConstr(total_overlap <= M * (1 - is_disjoint))
        model.addConstr(total_overlap >= 1 - M * is_disjoint)

        # Compute upper bounds
        for i in range(n_states):
            if i == s_prime:
                continue  # Excludes target state (s_prime) of LB calculation from sum.

            upper = model.addVar(lb=0, ub=1, name=f"ub_{i}")
            upper_cf_bounds[i] = upper

            # If the state-action pair is the observed, we know the CF probs exactly.
            if s == self.observed_state and a == self.observed_action:
                if i == self.observed_next_state:
                    model.addConstr(upper == 1.0)
                else:
                    model.addConstr(upper == 0.0)

            else:
                # This calculates what the upper bounds would be, if the distributions have disjoint support.
                # Disjoint: min(P_obs[s'], P_cf[i]) / P_obs[s']
                min_disjoint = model.addVar(lb=0, name=f"min_disjoint_{i}")
                model.addConstr(min_disjoint <= P_cf[i])
                model.addConstr(min_disjoint <= P_obs[self.observed_next_state])

                ratio_disjoint = model.addVar(lb=0, name=f"ratio_disjoint_{i}")
                model.addConstr(ratio_disjoint * P_obs[self.observed_next_state] == min_disjoint)
                
                ub_disjoint = model.addVar(lb=0, name=f"ub_disjoint_{i}")
                model.addConstr(ub_disjoint == ratio_disjoint)

                # This calculates what the upper bounds would be, if the distributions have overlapping support.
                ub_overlap = model.addVar(lb=0, name=f"ub_overlap_{i}")
                
                # If this is the observed next state, monotonicity holds, but not CS.
                # i == self.observed_next_state
                # ub = 1 if P_cf[i] ≥ P_obs[i],
                # ub = P_cf[i] / P_obs[i] otherwise
                # gamma=1 <-> P_obs[i] <= P_cf[i]
                if i == self.observed_next_state:
                    gamma = model.addVar(vtype=GRB.BINARY, name=f"gamma_{i}")
                    model.addConstr(P_obs[i] <= P_cf[i] + M * (1 - gamma))

                    model.addConstr(ub_overlap <= 1 + M * (1 - gamma))
                    model.addConstr(ub_overlap >= 1 - M * (1 - gamma))

                    ratio_var = model.addVar(lb=0, name=f"ratio_var_{i}")
                    model.addConstr(ratio_var * P_obs[i] == P_cf[i])

                    model.addConstr(ub_overlap <= ratio_var + M * gamma)
                    model.addConstr(ub_overlap >= ratio_var - M * gamma)

                # Otherwise, monotonicity may hold, and CS may hold.
                else:
                    # LHS of CS condition.
                    lhs = model.addVar(lb=0, name=f"lhs_{i}")
                    # RHS of CS condition.
                    rhs = model.addVar(lb=0, name=f"rhs_{i}")

                    # delta_i=1 <-> CS condition reduces CF prob to 0.
                    delta_i = model.addVar(vtype=GRB.BINARY, name=f"delta_{i}")

                    model.addConstr(lhs == P_cf[self.observed_next_state] * P_obs[i])
                    model.addConstr(rhs == P_cf[i] * P_obs[self.observed_next_state])
                    model.addConstr(lhs <= rhs + M * delta_i)

                    # CF prob = 0 if CS affects it.

                    # Case: P_obs[i] > 0 (has support)
                    ub_overlap_pos_obs = model.addVar(name=f"ub_pos_obs_{i}")
                    min1 = model.addVar(lb=0, name=f"min1_{i}")
                    model.addConstr(min1 <= 1 - P_cf[self.observed_next_state])
                    model.addConstr(min1 <= P_cf[i])

                    # Apply the correct logic:
                    # - If delta_i == 1 (CS fails): ub = 0
                    # - If delta_i == 0 (CS passes): ub = min(1 - P_cf[s'], P_cf[i])
                    model.addConstr(ub_overlap_pos_obs <= M * (1 - delta_i))        # Enforce ub = 0 when CS fails
                    model.addConstr(ub_overlap_pos_obs <= min1 + M * delta_i)       # Tight min when CS passes

                    # Case: P_obs[i] == 0 (no observed support)
                    ub_overlap_zero_obs = model.addVar(name=f"ub_zero_obs_{i}")
                    ratio = model.addVar(name=f"ratio_zero_obs_{i}")
                    model.addConstr(ratio * P_obs[self.observed_next_state] == P_cf[i])

                    min2 = model.addVar(lb=0, name=f"min2_{i}")
                    model.addConstr(min2 <= 1 - P_cf[self.observed_next_state])
                    model.addConstr(min2 <= ratio)
                    model.addConstr(ub_overlap_zero_obs == min2)

                    # Final blend: if support_obs[i] == 1 → use ub_overlap_pos_obs, else use ub_overlap_zero_obs
                    model.addConstr(
                        ub_overlap == ub_overlap_pos_obs * support_obs[i] + ub_overlap_zero_obs * (1 - support_obs[i])
                    )
                    
                # Select correct upper bound
                model.addConstr(upper <= ub_disjoint + M * (1 - is_disjoint))
                model.addConstr(upper >= ub_disjoint - M * (1 - is_disjoint))

                model.addConstr(upper <= ub_overlap + M * is_disjoint)
                model.addConstr(upper >= ub_overlap - M * is_disjoint)

        # Set objective
        model.setObjective(gp.quicksum(upper_cf_bounds[i] for i in range(n_states) if i != s_prime), GRB.MAXIMIZE)
        model.optimize()

        return model.ObjVal
    

    def lower_bound_tight(self, s, a, s_prime):
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
                return (self.imdp_transition_matrix[s, a, s_prime, 0] - (1 - self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0])) / self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]
            else:
                return 0.0 
 
        # Otherwise, the supports overlap.

        global _worker_env
        model = gp.Model(env=_worker_env)

        # Case 1: s' = s_{t+1}
        if s_prime == self.observed_next_state:
            result = max(self.imdp_transition_matrix[s, a, s_prime, 0], 1 - self._optimise_sum_max_ubs_of_other_transitions(model, s, a, s_prime))
        
        # Case 2: counterfactual stability could limit the CF prob to 0.
        elif self._sometimes_counterfactual_stability(s, a, s_prime):
            result = 0
        
        # Case 3: all other cases.
        result = max(0, 1 - self._optimise_sum_max_ubs_of_other_transitions(model, s, a, s_prime))
        model.dispose()

        return result


    def lower_bound_approx(self, s, a, s_prime):
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
                return (self.imdp_transition_matrix[s, a, s_prime, 0] - (1 - self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0])) / self.imdp_transition_matrix[self.observed_state, self.observed_action, self.observed_next_state, 0]
            else:
                return 0.0 
 
        # Otherwise, the supports overlap.
        
        # This implementation is non-tight, this sum produces a non-tight LB. But, we could replace this calculation with an optimisation problem to find the exact LB.
        def _sum_max_ubs_of_other_transitions():
            n_states = self.imdp_transition_matrix.shape[0]
            sum_ubs = 0.0

            for s_prime_prime in range(n_states):
                if s_prime != s_prime_prime:
                    sum_ubs += self.upper_bound(s, a, s_prime_prime)
            return sum_ubs

        # Case 1: s' = s_{t+1}
        if s_prime == self.observed_next_state:
            return max(self.imdp_transition_matrix[s, a, s_prime, 0], 1 - _sum_max_ubs_of_other_transitions())
        
        # Case 2: counterfactual stability could limit the CF prob to 0.
        elif self._sometimes_counterfactual_stability(s, a, s_prime):
            return 0
        
        # Case 3: all other cases.
        return max(0, 1 - _sum_max_ubs_of_other_transitions())
            

    def upper_bound(self, s, a, s_prime):
        if s == self.observed_state and a == self.observed_action:
            if s_prime == self.observed_next_state:
                return 1.0
            else:
                return 0.0
                        
        support_of_observed = self._get_support(self.observed_state, self.observed_action)
        support_of_cf = self._get_support(s, a)
        overlapping_support = support_of_observed.intersection(support_of_cf)

        assert(self.imdp_transition_matrix[s, a, s_prime, 1] >= 0.0)

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
    

    def calculate_all_bounds_tight(self, n_jobs=32):
        n_states = self.imdp_transition_matrix.shape[0]
        n_actions = self.imdp_transition_matrix.shape[1]

        interval_cf_transition_matrix = np.zeros(shape=(n_states, n_actions, n_states, 2), dtype=np.float64)

        # Only search through possible transitions.
        transitions = self.mdp.get_possible_transitions()

        def chunkify(lst, n):
            """Split list `lst` into `n` roughly equal chunks"""
            k, m = divmod(len(lst), n)
            return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

        def compute_chunk(chunk, lower_bound_approx, upper_bound):
            results = []
            for s, a, s_prime in chunk:
                lb = Decimal(lower_bound_approx(s, a, s_prime))
                assert not math.isnan(lb)
                # Remove small floating-point errors.
                lb = max(0.0, lb)
                lb = min(1.0, lb)

                ub = Decimal(upper_bound(s, a, s_prime))
                assert not math.isnan(ub)
                # Remove small floating-point errors.
                ub = max(0.0, ub)
                ub = min(1.0, ub)

                results.append((s, a, s_prime, lb, ub))

            return results

        # Split into chunks
        chunks = chunkify(transitions, n_jobs)

        # Compute chunks in parallel
        chunked_results = Parallel(
            n_jobs=n_jobs,           # number of parallel workers
            backend="loky",          # backend
            initializer=_init_worker # optional initializer function
        )(
            delayed(compute_chunk)(chunk, self.lower_bound_tight, self.upper_bound)
            for chunk in chunks
        )
            
        # Flatten results
        results = [item for sublist in chunked_results for item in sublist]

        # Fill the matrix
        interval_cf_transition_matrix = np.zeros((n_states, n_actions, n_states, 2), dtype=np.float64)

        for s, a, s_prime, lb, ub in results:
            interval_cf_transition_matrix[s, a, s_prime, 0] = lb
            interval_cf_transition_matrix[s, a, s_prime, 1] = ub

        return interval_cf_transition_matrix

    
    def calculate_all_bounds_approx(self, n_jobs=32):
        n_states = self.imdp_transition_matrix.shape[0]
        n_actions = self.imdp_transition_matrix.shape[1]

        # Only search through the possible transitions.
        transitions = self.mdp.get_possible_transitions()

        def chunkify(lst, n):
            """Split list `lst` into `n` roughly equal chunks"""
            k, m = divmod(len(lst), n)
            return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

        def compute_chunk(chunk, lower_bound_approx, upper_bound):
            results = []
            for s, a, s_prime in chunk:
                lb = Decimal(lower_bound_approx(s, a, s_prime))
                assert not math.isnan(lb)
                
                # Remove small floating-point errors.
                lb = max(0.0, lb)
                lb = min(lb, 1.0)

                ub = Decimal(upper_bound(s, a, s_prime))
                assert not math.isnan(ub)
                
                # Remove small floating-point errors.
                ub = max(0.0, ub)
                ub = min(1.0, ub)

                if not ub >= lb:
                    def _sum_max_ubs_of_other_transitions():
                        n_states = self.imdp_transition_matrix.shape[0]
                        sum_ubs = 0.0

                        for s_prime_prime in range(n_states):
                            if s_prime != s_prime_prime:
                                sum_ubs += self.upper_bound(s, a, s_prime_prime)
                        return sum_ubs

                assert(ub >= lb)

                results.append((s, a, s_prime, lb, ub))

            return results

        # Split into chunks
        chunks = chunkify(transitions, n_jobs)

        # Compute chunks in parallel
        chunked_results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(compute_chunk)(chunk, self.lower_bound_approx, self.upper_bound)
            for chunk in chunks
        )

        # Flatten results
        results = [item for sublist in chunked_results for item in sublist]

        # Fill the matrix
        interval_cf_transition_matrix = np.zeros((n_states, n_actions, n_states, 2), dtype=np.float64)

        for s, a, s_prime, lb, ub in results:
            interval_cf_transition_matrix[s, a, s_prime, 0] = lb
            interval_cf_transition_matrix[s, a, s_prime, 1] = ub

        return interval_cf_transition_matrix


    def calculate_upper_bounds(self, chunk, upper_bounds):
        for (s, a, s_prime) in chunk:
            ub = Decimal(self.upper_bound(s, a, s_prime))
            upper_bounds[(s, a, s_prime)] = ub


    def calculate_lower_bounds_tight(self, chunk, lower_bounds):
        for (s, a, s_prime) in chunk:
            lb = Decimal(self.lower_bound_tight(s, a, s_prime))
            lower_bounds[(s, a, s_prime)] = lb

    def calculate_lower_bounds_approx(self, chunk, lower_bounds):
        for (s, a, s_prime) in chunk:
            lb = Decimal(self.lower_bound_approx(s, a, s_prime))
            lower_bounds[(s, a, s_prime)] = lb
    

class MultiStepCFBoundCalculatorTight:
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
            bound_calculator = CFBoundCalculator(trajectory[t][0], trajectory[t][2], trajectory[t][1], self.imdp_transition_matrix)
            interval_cf_transition_matrix[t] = bound_calculator.calculate_all_bounds_tight()

        return interval_cf_transition_matrix
    

class MultiStepCFBoundCalculatorApprox:
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
            bound_calculator = CFBoundCalculator(trajectory[t][0], trajectory[t][2], trajectory[t][1], self.imdp_transition_matrix)
            interval_cf_transition_matrix[t] = bound_calculator.calculate_all_bounds_approx()

        return interval_cf_transition_matrix


class ParallelMultiStepCFBoundCalculatorApprox:
    def __init__(self, mdp, imdp_transition_matrix):
        self.mdp = mdp
        self.imdp_transition_matrix = imdp_transition_matrix
    
    def calculate_bounds(self, trajectory, n_jobs=32):
        n_timesteps = len(trajectory)
        n_states = self.imdp_transition_matrix.shape[0]
        n_actions = self.imdp_transition_matrix.shape[1]
        assert self.imdp_transition_matrix.shape[2] == n_states
        assert self.imdp_transition_matrix.shape[3] == 2

        interval_cf_transition_matrix = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states, 2))

        for t in range(n_timesteps):
            bound_calculator = CFBoundCalculator(self.mdp, trajectory[t][0], trajectory[t][2], trajectory[t][1], self.imdp_transition_matrix)
            bounds_t = bound_calculator.calculate_all_bounds_approx(n_jobs=n_jobs)
            interval_cf_transition_matrix[t] = bounds_t

        return interval_cf_transition_matrix

class ParallelMultiStepCFBoundCalculatorTight:
    def __init__(self, mdp, imdp_transition_matrix):
        self.mdp = mdp
        self.imdp_transition_matrix = imdp_transition_matrix
    

    def calculate_bounds(self, trajectory, n_jobs=32):
        n_timesteps = len(trajectory)
        n_states = self.imdp_transition_matrix.shape[0]
        n_actions = self.imdp_transition_matrix.shape[1]
        assert self.imdp_transition_matrix.shape[2] == n_states
        assert self.imdp_transition_matrix.shape[3] == 2

        interval_cf_transition_matrix = np.zeros(shape=(n_timesteps, n_states, n_actions, n_states, 2))

        for t in range(n_timesteps):
            bound_calculator = CFBoundCalculator(self.mdp, trajectory[t][0], trajectory[t][2], trajectory[t][1], self.imdp_transition_matrix)
            bounds_t = bound_calculator.calculate_all_bounds_tight(n_jobs=n_jobs)
            interval_cf_transition_matrix[t] = bounds_t

        return interval_cf_transition_matrix
    