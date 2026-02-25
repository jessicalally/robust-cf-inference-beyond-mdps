import cf.counterfactual as cf
from .mdp import MDP
import numpy as np
import pickle
import random
from scipy.linalg import block_diag


class Action(object):
    NUM_ACTIONS_TOTAL = 8
    ANTIBIOTIC_STRING = "antibiotic"
    VENT_STRING = "ventilation"
    VASO_STRING = "vasopressors"
    ACTION_VEC_SIZE = 3

    def __init__(self, selected_actions = None, action_idx = None):
        
        # Actions can be specified in two ways: by providing a list of selected actions (as strings) or by an action index.
        assert (selected_actions is not None and action_idx is None) \
            or (selected_actions is None and action_idx is not None), \
            "must specify either set of action strings or action index"
            
        if selected_actions is not None:
            if Action.ANTIBIOTIC_STRING in selected_actions:
                self.antibiotic = 1
            else:
                self.antibiotic = 0
            if Action.VENT_STRING in selected_actions:
                self.ventilation = 1
            else:
                self.ventilation = 0
            if Action.VASO_STRING in selected_actions:
                self.vasopressors = 1
            else:
                self.vasopressors = 0
                
        else:
            mod_idx = action_idx
            term_base = Action.NUM_ACTIONS_TOTAL/2
            self.antibiotic = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2
            self.ventilation = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2
            self.vasopressors = np.floor(mod_idx/term_base).astype(int)
            
            '''
            There are three treatments (A, E, V) and thus 2^3 = 8 possible action combinations. 
            The binary representation of action_idx from 0 to 7 can be thought of as the action combinations:

                000 -> No treatments
                001 -> V
                010 -> E
                011 -> E, V
                100 -> A
                101 -> A, V
                110 -> A, E
                111 -> A, E, V
                
            The code block breaks down action_idx to understand which treatments are being used and initializes the three attributes (self.antibiotic, self.ventilation, self.vasopressors) accordingly.
            '''
            
    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.antibiotic == other.antibiotic and \
            self.ventilation == other.ventilation and \
            self.vasopressors == other.vasopressors

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def get_action_idx(self):
        assert self.antibiotic in (0, 1)
        assert self.ventilation in (0, 1)
        assert self.vasopressors in (0, 1)
        return 4*self.antibiotic + 2*self.ventilation + self.vasopressors
    '''
    The weighted sum effectively encodes the three binary values into a single integer (form 0 to 7; NUM_ACTIONS_TOTAL = 8 in total). 
    The weights (4, 2, and 1) were chosen to uniquely identify each combination of the three treatments.
    
    For example:

        If only antibiotic is used: action_idx = 4*1 + 2*0 + 0*1 = 4.
        If only ventilation is used: action_idx = 4*0 + 2*1 + 0*1 = 2.
        If antibiotic and ventilation are used: action_idx = 4*1 + 2*1 + 0*1 = 6.
        If all three are used: action_idx = 4*1 + 2*1 + 1*1 = 7.
    '''

    def __hash__(self):
        return self.get_action_idx()
    
    def get_selected_actions(self):
        selected_actions = set()
        if self.antibiotic == 1:
            selected_actions.add(Action.ANTIBIOTIC_STRING)
        if self.ventilation == 1:
            selected_actions.add(Action.VENT_STRING)
        if self.vasopressors == 1:
            selected_actions.add(Action.VASO_STRING)
        return selected_actions
    
    def get_abbrev_string(self):
        '''
        AEV: antibiotics, ventilation, vasopressors
        '''
        output_str = ''
        if self.antibiotic == 1:
            output_str += 'A'
        if self.ventilation == 1:
            output_str += 'E'
        if self.vasopressors == 1:
            output_str += 'V'
        return output_str
    
    def get_action_vec(self):
        return np.array([[self.antibiotic], [self.ventilation], [self.vasopressors]])


class State(object):
    NUM_OBS_STATES = 720
    NUM_HID_STATES = 2  # Diabetic status is hidden.
    NUM_PROJ_OBS_STATES = int(720 / 5)  # Marginalising over glucose
    NUM_FULL_STATES = int(NUM_OBS_STATES * NUM_HID_STATES)

    def __init__(self, state_idx = None, idx_type = 'obs', diabetic_idx = None, state_categs = None):
        # Initialises the state either by its index or by passing specific categories for each state variable.
        assert state_idx is not None or state_categs is not None
        assert ((diabetic_idx is not None and diabetic_idx in [0, 1]) or
                (state_idx is not None and idx_type == 'full'))

        assert idx_type in ['obs', 'full', 'proj_obs']

        if state_idx is not None:
            self.set_state_by_idx(
                    state_idx, idx_type=idx_type, diabetic_idx=diabetic_idx)
        elif state_categs is not None:
            assert len(state_categs) == 7, "must specify 7 state variables"
            self.hr_state = state_categs[0]
            self.sysbp_state = state_categs[1]
            self.percoxyg_state = state_categs[2]
            self.glucose_state = state_categs[3]
            self.antibiotic_state = state_categs[4]
            self.vaso_state = state_categs[5]
            self.vent_state = state_categs[6]
            self.diabetic_idx = diabetic_idx

    def check_absorbing_state(self):
        num_abnormal = self.get_num_abnormal()
        if num_abnormal >= 3:
            return True
        elif num_abnormal == 0 and not self.on_treatment():
            return True
        return False
    
    def state_rewards(self):
        num_abnormal = self.get_num_abnormal()
        if num_abnormal >= 3:
            return (-1000)
        elif num_abnormal == 2:
            return (-50)
        elif num_abnormal == 1:
            return (+50)
        elif num_abnormal == 0 and self.on_treatment():
            return (+70)
        elif num_abnormal == 0 and not self.on_treatment():
            return (+1000)

    def set_state_by_idx(self, state_idx, idx_type, diabetic_idx=None):
        """set_state_by_idx

        The state index is determined by using "bit" arithmetic, with the
        complication that not every state is binary

        :param state_idx: Given index
        :param idx_type: Index type, either observed (720), projected (144) or
        full (1440)
        :param diabetic_idx: If full state index not given, this is required
        """
        
        if idx_type == 'obs':
            term_base = State.NUM_OBS_STATES/3
        elif idx_type == 'proj_obs':
            term_base = State.NUM_PROJ_OBS_STATES/3
        elif idx_type == 'full':
            term_base = State.NUM_FULL_STATES/2
        
        mod_idx = state_idx

        if idx_type == 'full':           
            self.diabetic_idx = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 3
        else:
            assert diabetic_idx is not None
            self.diabetic_idx = diabetic_idx

        self.hr_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 3
        self.sysbp_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.percoxyg_state = np.floor(mod_idx/term_base).astype(int)

        if idx_type == 'proj_obs':
            self.glucose_state = 2
        else:
            mod_idx %= term_base
            term_base /= 5
            self.glucose_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.antibiotic_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.vaso_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.vent_state = np.floor(mod_idx/term_base).astype(int)


    def get_state_idx(self, idx_type='obs'):
        '''
        returns integer index of state: significance order as in categorical array
        '''
        
        if idx_type == 'obs':
            categ_num = np.array([3,3,2,5,2,2,2])
            state_categs = [
                    self.hr_state,
                    self.sysbp_state,
                    self.percoxyg_state,
                    self.glucose_state,
                    self.antibiotic_state,
                    self.vaso_state,
                    self.vent_state]
        elif idx_type == 'proj_obs':
            categ_num = np.array([3,3,2,2,2,2])
            state_categs = [
                    self.hr_state,
                    self.sysbp_state,
                    self.percoxyg_state,
                    self.antibiotic_state,
                    self.vaso_state,
                    self.vent_state]
        elif idx_type == 'full':
            categ_num = np.array([2,3,3,2,5,2,2,2])
            state_categs = [
                    self.diabetic_idx,
                    self.hr_state,
                    self.sysbp_state,
                    self.percoxyg_state,
                    self.glucose_state,
                    self.antibiotic_state,
                    self.vaso_state,
                    self.vent_state]

        sum_idx = 0
        prev_base = 1
        for i in range(len(state_categs)):
            idx = len(state_categs) - 1 - i
            sum_idx += prev_base*state_categs[idx]
            prev_base *= categ_num[idx]
        return sum_idx
    
    def __eq__(self, other):
        '''
        override equals: two states equal if all internal states same
        '''
        return isinstance(other, self.__class__) and \
            self.hr_state == other.hr_state and \
            self.sysbp_state == other.sysbp_state and \
            self.percoxyg_state == other.percoxyg_state and \
            self.glucose_state == other.glucose_state and \
            self.antibiotic_state == other.antibiotic_state and \
            self.vaso_state == other.vaso_state and \
            self.vent_state == other.vent_state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.get_state_idx()

    def get_num_abnormal(self):
        '''
        returns number of abnormal conditions
        '''
        num_abnormal = 0
        if self.hr_state != 1:
            num_abnormal += 1
        if self.sysbp_state != 1:
            num_abnormal += 1
        if self.percoxyg_state != 1:
            num_abnormal += 1
        if self.glucose_state != 2:
            num_abnormal += 1
        return num_abnormal
    
    def on_treatment(self):
        '''
        returns True iff any of 3 treatments active
        '''
        if self.antibiotic_state == 0 and \
            self.vaso_state == 0 and self.vent_state == 0:
            return False
        return True

    def on_antibiotics(self):
        '''
        returns True iff antibiotics active
        '''
        return self.antibiotic_state == 1

    def on_vasopressors(self):
        '''
        returns True iff vasopressors active
        '''
        return self.vaso_state == 1

    def on_ventilation(self):
        '''
        returns True iff ventilation active
        '''
        return self.vent_state == 1

    def copy_state(self):
        return State(state_categs = [
            self.hr_state,
            self.sysbp_state,
            self.percoxyg_state,
            self.glucose_state,
            self.antibiotic_state,
            self.vaso_state,
            self.vent_state],
            diabetic_idx=self.diabetic_idx)

    def get_state_vector(self):
        return np.array([self.hr_state,
            self.sysbp_state,
            self.percoxyg_state,
            self.glucose_state,
            self.antibiotic_state,
            self.vaso_state,
            self.vent_state]).astype(int)
    

class SepsisMDP(MDP):
    def __init__(self):
        self.env_name = "sepsis"
        self.init_state = 1348
        self.states = range(1440)
        self.actions = range(8)
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.sink_state = np.full(self.n_states, False)
        self.transition_matrix, self.rewards, self.state_rewards = self.load_data()
        self.optimal_policy = np.zeros(self.n_states, dtype=int)
        self.values = np.zeros(self.n_states)
        self.valid_actions = self._extract_valid_actions()
        self.max_reward = np.max(self.state_rewards)
        
        # Hyperparameters
        self.discount = 0.9

        self.value_iteration()


    def set_suboptimal_trajectory(self, suboptimal_trajectory):
        self.suboptimal_trajectory = suboptimal_trajectory
        self.init_state = self.suboptimal_trajectory[0][0]

    
    def set_optimal_trajectory(self, optimal_trajectory):
        self.optimal_trajectory = optimal_trajectory

    
    def load_data(self):
        with open("data/diab_txr_mats-replication.pkl", "rb") as f:
            mdict = pickle.load(f)

        tx_mat = mdict["tx_mat"]
        r_mat = mdict["r_mat"]

        tx_mat_full = np.zeros((len(self.actions), State.NUM_FULL_STATES, State.NUM_FULL_STATES))
        r_mat_full = np.zeros((len(self.actions), State.NUM_FULL_STATES, State.NUM_FULL_STATES))
        
        for a in range(len(self.actions)):
            tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a,...])
            r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])

        # Modify transition probabilities and rewards to account for absorbing states.
        all_absorbing_states = []
        all_absorbing_rewards = []
        self.non_absorbing_states = []
        all_rewards = []

        for s in range(1440):
            get_states = State(state_idx=s, idx_type = 'full')
            abs = get_states.check_absorbing_state()
            if abs == True: 
                all_absorbing_states.append(s)
                rew = get_states.state_rewards()
                all_absorbing_rewards.append(rew)
                
            if abs == False:
                self.non_absorbing_states.append(s)

            rew = get_states.state_rewards()
            all_rewards.append(rew)

        for s in range(1440):
            for a in range(8):
                if s in all_absorbing_states:
                    tx_mat_full[a, s, :] = np.zeros(1440)
                    tx_mat_full[a, s, s] = 1 

        for s in range(1440):
            for a in range(8):
                if s in all_absorbing_states:
                    reward_idx = all_absorbing_states.index(s)
                    r_mat_full[a, s, :] = np.full((1440,), (all_absorbing_rewards[reward_idx]))
                else:
                    for s_p in np.where(tx_mat_full[a, s, :]!=0)[0]:
                        r_mat_full[a, s, s_p] = all_rewards[s_p]

        for s in all_absorbing_states:
            self.sink_state[s] = True

        rewards_pi = np.zeros((1440, 8)) 

        for s in range(1440):
            for a in range(8):
                s_p = (np.where(tx_mat_full[a, s, :] == (np.max(tx_mat_full[a, s, :]))))[0][0]
                rewards_pi[s, a] = r_mat_full[a, s, s_p]

        transition_matrix = np.swapaxes(tx_mat_full, 0, 1)
        self.tx_mat_full = tx_mat_full
        self.r_mat_full = r_mat_full

        return transition_matrix, rewards_pi, all_rewards


    # Samples a random trajectory from a suboptimal policy.
    def sample_random_trajectory(self, n_steps=10):
        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = random.choice(self.non_absorbing_states)

        for time_idx in range(n_steps):
            action = np.random.choice(8, size=1, p=self.suboptimal_policy[current_state])[0]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0] 
            reward = self.state_rewards[current_state]
            trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        return trajectory.astype(int)
    

    # Samples trajectory produced by optimal policy.
    def sample_optimal_trajectory(self, n_steps=10):
        n_state = 4
        trajectory = np.zeros((n_steps, n_state))

        current_state = random.choice(self.non_absorbing_states)

        for time_idx in range(n_steps):
            action = np.random.choice(8, size=1, p=self.optimal_policy[current_state])[0]
            next_state = np.random.choice(len(self.states), size=1, p=self.transition_matrix[current_state, action, :])[0] 
            reward = self.state_rewards[current_state]
            trajectory[time_idx, :] = np.array([current_state, next_state, action, reward])
            current_state = next_state

        return trajectory.astype(int)
    

    # Returns an example suboptimal trajectory that enters a dangerous, terminal state.
    def sample_suboptimal_trajectory(self, n_steps=10):
        return np.array([[ 1348,  1337,     2,   -50],
             [ 1337,  1348,     4,    50],
             [ 1348,  1416,     0,   -50],
             [ 1416,  1408,     0,   -50],
             [ 1408,  1408,     5, -1000],
             [ 1408,  1408,     4, -1000],
             [ 1408,  1408,     6, -1000],
             [ 1408,  1408,     5, -1000],
             [ 1408,  1408,     0, -1000],
             [ 1408,  1408,     6, -1000]]).astype(int)


    def _extract_valid_actions(self):
        return np.full(shape=(self.n_states, self.n_actions), fill_value=True)
