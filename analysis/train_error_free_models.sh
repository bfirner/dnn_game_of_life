#! /bin/bash

# The paper plotted success vs m_factor for different depths

step=2
m_factor=
d_factor=2

for step in `seq 2 3`; do
    for m_factor in `seq 20 10 40`; do
        for d_factor in 1 1.5 2; do
            echo "Attempting to find successful model for ${step} ${m_factor} ${d_factor}"
            # Keep training models until one is successful.
            succ=`python3 ~/projects/gameoflife/gameoflife.py --step $step --m_factor $m_factor\
                --d_factor $d_factor  --use_sigmoid  --activation_fun LeakyReLU --normalize  \
                --batches 25000 --batch_size 128 | tail -n 1 | cut --d " " -f 2`
            while [ "$succ" != "success." ]; do
                succ=`python3 ~/projects/gameoflife/gameoflife.py --step $step --m_factor $m_factor\
                    --d_factor $d_factor  --use_sigmoid  --activation_fun LeakyReLU --normalize  \
                    --batches 25000 --batch_size 128 | tail -n 1 | cut --d " " -f 2`
            done
            cp "gol_model.pyt" "gol_model_${step}_${m_factor}_${d_factor}.pyt"
        done
    done
done

