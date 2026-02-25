import argparse
from evaluation import *
import environments
import h5py
import itertools
import math
from evaluation import value_functions
from src.utils import load_generated_policy, load_value_function, load_value_function_max_actions
import os
import numpy as np
from pathlib import Path
import dill as pickle

# Returns the MDP environment.
def get_mdp_env(env_name):
    env_map = {
        "gridworld": environments.GridWorldMDP(),
        "gridworld-uncertain": environments.GridWorldMDP(p_r=0.4),
        "aircraft_tiny": environments.AircraftMDP(),
        "sepsis": environments.SepsisMDP(),
        "frozen_lake": environments.FrozenLakeMDP(),
    }
    
    try:
        return env_map[env_name]
    except KeyError:
        raise ValueError(f"Unknown environment name: {env_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(f"--o", type=str, default="generate_data_max_actions")
    parser.add_argument(f"--env", type=str, default="gridworld")
    parser.add_argument(f"--max_t", type=int, default=10)
    parser.add_argument(f"--init", type=int, default=0)
    args = parser.parse_args()
    
    function_name = args.o
    env_name = args.env
    init_state = args.init

    if not os.path.exists(f"MDPs/{env_name}/pac"):
        os.makedirs(f"MDPs/{env_name}/pac")

    if not os.path.exists(f"ICFMDPs/{env_name}/pac"):
        os.makedirs(f"ICFMDPs/{env_name}/pac")

    if not os.path.exists(f"results/{env_name}/pac"):
        os.makedirs(f"results/{env_name}/pac")

    if Path(f"data/{env_name}.pickle").is_file():
        with open(f"data/{env_name}.pickle", "rb") as f:
            mdp = pickle.load(f)
    else:
        # Get MDP environment.
        mdp = get_mdp_env(env_name)

    with open(f"data/{env_name}.pickle", "wb") as f:
        pickle.dump(mdp, f)

    # Define hyperparameters.
    possible_gammas = [0.01, 0.05, 0.1, 0.2]
    
    # Sepsis requires 10^7 trajectories to be very accurate.
    possible_trajectories = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]

    hyperparameters = list(itertools.product(possible_trajectories, possible_gammas))
    
    if function_name == "generate_data_max_actions":
        np.random.seed(1)
        generate_data_max_actions_experiment(mdp, args.max_t, args.init)

    elif function_name == "run_max_actions":
        np.random.seed(1)
        run_max_actions_experiment(mdp, args.max_t, args.init)

    elif function_name == "cf_correctness":
        np.random.seed(1)
        check_cf_correctness(mdp, args.max_t, [1000, 10000, 100000, 1000000, 10000000], [0.1])
    else:
       print(f"Function '{function_name}' is not recognized.")

main()
