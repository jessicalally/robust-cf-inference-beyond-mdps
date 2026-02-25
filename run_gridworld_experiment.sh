#!/bin/bash

echo "GridWorld (p=0.9)"
echo "Generating the ICFMDPs..."
python3 evaluation.py --env gridworld --o generate_data_max_actions --max_t 10 --init 0

# Generate policies from Julia files.
for i in {0..99}; do
    for max_transitions in 100000; do
        for delta in 0.1; do
            echo "Running experiments $i"
            julia MDPs/GridWorld_0.9/pac/simulated_GridWorld_0.9_${max_transitions}_${delta}_max_actions_${i}_approx.jl
            julia MDPs/GridWorld_0.9/pac/simulated_GridWorld_0.9_${max_transitions}_${delta}_max_actions_${i}_tight.jl
            julia MDPs/GridWorld_0.9/pac/simulated_GridWorld_0.9_${max_transitions}_${delta}_max_actions_${i}_gumbel.jl
        done
    done

    julia MDPs/GridWorld_0.9/pac/simulated_GridWorld_0.9_true_max_actions_${i}.jl
done

echo "Evaluating policies..."
python3 evaluation.py --env gridworld --o run_max_actions --max_t 10 --init 0

# Evaluating CF correctness.
echo "Evaluating CF correctness..."
python3 evaluation.py --env gridworld --o cf_correctness --max_t 10 >> gridworld_0.9_simulated_cf_correctness.txt
