#!/bin/bash

echo "Aircraft"
echo "Generating the ICFMDPs..."
python3 evaluation.py --env aircraft_tiny --o generate_data_max_actions --max_t 12 --init 0

# Generate policies from Julia files.
for i in {0..99}; do
    for max_transitions in 100000; do
        for delta in 0.1; do
            echo "Running experiments $i"
            julia MDPs/aircraft_tiny/pac/simulated_aircraft_tiny_${max_transitions}_${delta}_max_actions_${i}_approx.jl
            julia MDPs/aircraft_tiny/pac/simulated_aircraft_tiny_${max_transitions}_${delta}_max_actions_${i}_tight.jl
        done
    done

    julia MDPs/aircraft_tiny/pac/simulated_aircraft_tiny_true_max_actions_${i}.jl
done

echo "Evaluating policies..."
python3 evaluation.py --env aircraft_tiny --o run_max_actions --max_t 12 --init 0