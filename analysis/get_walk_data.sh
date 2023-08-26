#! /bin/bash

# Get the walk error for each of the models.

for step in `seq 2 3`; do
    for m_factor in `seq 10 10 40`; do
        for d_factor in 1 1.5 2; do
            echo "Generating walks for ${step} ${m_factor} ${d_factor}"
            for trial in `seq 1 10`; do
                python3 ~/projects/gameoflife/loss_walk.py --step $step --m_factor $m_factor \
                    --d_factor $d_factor --use_sigmoid --activation_fun LeakyReLU --normalize \
                    --batches 1000 --batch_size 128 \
                    --resume_from models/gol_model_${step}_${m_factor}_${d_factor}.pyt > \
                    error_walk_${step}_${m_factor}_${d_factor}_${trial}.txt
            done
        done
    done
done

for step in `seq 2 3`; do
    for m_factor in 1; do
        for d_factor in 1; do
            echo "Generating walks for ${step} ${m_factor} ${d_factor}"
            for trial in `seq 1 10`; do
                python3 ~/projects/gameoflife/loss_walk.py --step $step --m_factor $m_factor \
                    --d_factor $d_factor --use_sigmoid --activation_fun LeakyReLU \
                    --batches 1000 --batch_size 128 \
                    --resume_from models/gol_model_${step}_${m_factor}_${d_factor}.pyt > \
                    straight_walk_results/error_walk_${step}_${m_factor}_${d_factor}_${trial}.txt
            done
        done
    done
done
