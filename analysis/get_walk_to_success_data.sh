#! /bin/bash

# Get the walk error for each bad model to the good model.

for step in `seq 1 3` 5 10 20; do
    for m_factor in 1 5 10 50 100 200; do
        for d_factor in 1.0; do
            # If a model was trained with these settings then generate a walk for it.
            if [[ -f "models/gol_model_steps${step}_m${m_factor}_d${d_factor}_seed1.pyt" ]];
            then
                # First, create a "correct" model.
                python3 ../gameoflife.py --step $step --m_factor $m_factor \
                    --d_factor $d_factor  --use_sigmoid  --activation_fun LeakyReLU --presolve  \
                    --batches 1000 --batch_size 128 --seed 100 --use_cuda
                for seed in `seq 1 10`; do
                    python3 ../loss_walk.py --step $step --batches 1000 \
                    --use_sigmoid --m_factor $m_factor --activation_fun LeakyReLU --batch_size 128 --destination_weights \
                    "gol_model_presolve_steps${step}_m${m_factor}_d${d_factor}_seed100.pyt" --use_cuda --resume_from \
                    "nonorm_models/gol_model_steps${step}_m${m_factor}_d${d_factor}_seed${seed}.pyt" \
                    > bad_to_good_walk_results/error_walk_steps${step}_m${m_factor}_d${d_factor}_seed${seed}.txt
                done
            fi
        done
    done
done


