#!/bin/bash

echo "Frozen Lake"
echo "Generating the ICFMDPs..."
python3 evaluation.py --env frozen_lake --o generate_data_max_actions --max_t 20 --init 0

# Generate policies from Julia files.
for i in {0..99}; do
    for max_transitions in 100000; do
        for delta in 0.1; do
            echo "Running experiments $i"
            julia MDPs/frozen_lake/pac/simulated_frozen_lake_${max_transitions}_${delta}_max_actions_${i}_approx.jl
            julia MDPs/frozen_lake/pac/simulated_frozen_lake_${max_transitions}_${delta}_max_actions_${i}_tight.jl
            julia MDPs/frozen_lake/pac/simulated_frozen_lake_${max_transitions}_${delta}_max_actions_${i}_gumbel.jl
        done
    done

    julia MDPs/frozen_lake/pac/simulated_frozen_lake_true_max_actions_${i}.jl
done

echo "Evaluating policies..."
python3 evaluation.py --env frozen_lake --o run_max_actions --max_t 20 --init 0