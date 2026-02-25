#!/bin/bash

echo "Sepsis"
echo "Generating the ICFMDPs..."
python3 evaluation.py --env sepsis --o generate_data_max_actions --max_t 10 --init 1348

# Generate policies from Julia files.
for i in {0..99}; do
    for max_transitions in 100000; do
        for delta in 0.1; do
            echo "Running experiments $i"
            julia MDPs/sepsis/pac/simulated_sepsis_${max_transitions}_${delta}_max_actions_${i}_approx.jl
            julia MDPs/sepsis/pac/simulated_sepsis_${max_transitions}_${delta}_max_actions_${i}_tight.jl
        done
    done

    julia MDPs/sepsis/pac/simulated_sepsis_true_max_actions_${i}.jl
done
