#! /bin/bash

# Get the walk error for each bad model to the good model.

for seed in `seq 2 11`; do
    python3 ~/projects/gameoflife/loss_walk.py --step 2 --batches 15000 --use_sigmoid --m_factor 1 \
    --activation_fun LeakyReLU --batch_size 128 --destination_weights \
    ~/Documents/writing/presentations/images/game_of_life/models/gol_model_2_1_1.pyt --resume_from \
    ~/Documents/writing/presentations/images/game_of_life/failure_models/gol_model_2_1_1_seed_${seed}.pyt \
    > bad_to_good_walk_results/error_walk_seed_${seed}.txt
done
