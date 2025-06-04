#!/bin/bash

# Generating the ICFMDPs
echo "Generating the ICFMDPs..."
python3 evaluate_simulated_imdp.py train

# Run robust value iteration on generated ICFMDPs
echo "Running robust value iteration..."
for max_transitions in 100 500 1000 10000; do
    for delta in 0.01 0.05 0.1 0.2; do
        echo "Running experiment (${max_transitions}, ${delta})"
        julia "MDPs/simulated_gridworld_${max_transitions}_${delta}.jl"
    done
done

julia "MDPs/true_gridworld.jl"

# Evaluating policy performance
echo "Evaluating policy performance..."
python3 evaluate_simulated_imdp.py test

python3 evaluate_simulated_imdp.py foo