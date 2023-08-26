#! /bin/bash

# The paper plotted success vs m_factor and d_factor for different depths

echo "Steps M D SuccessRate AvgSuccess"

for step in `seq 1 3`; do
    for m_factor in 5 `seq 10 10 40`; do
        for d_factor in 1 1.5 2; do
            successes=0
            rates=""
            avg_rate=0
            for trial in `seq 20`; do
                out=`python3 ../gameoflife.py --step $step --m_factor $m_factor\
                    --d_factor $d_factor  --use_sigmoid  --activation_fun LeakyReLU --normalize  \
                    --batches 25000 --batch_size 128 --seed $trial | tail -n 2`
                succ=`echo "$out" | tail -n 1 | cut --d " " -f 2`
                rate=`echo "$out" | head -n 1 | cut --d " " -f 3`

                if [[ "$succ" = "success." ]]; then
                    successes=`echo $successes + 0.05 | bc -l`;
                fi

                rates=`echo "$rates $rate"`
                avg_rate=`echo "$avg_rate + $rate/20.0" | bc -l`
            done;
            echo "$step $m_factor $d_factor $successes $avg_rate $rates"
        done
    done

    for m_factor in 1; do
        for d_factor in 1; do
            successes=0
            rates=""
            avg_rate=0
            trial=1
            out=`python3 ../gameoflife.py --step $step --m_factor $m_factor\
                --d_factor $d_factor  --use_sigmoid  --activation_fun LeakyReLU --normalize  \
                --batches 25000 --batch_size 128 --seed $trial | tail -n 2`
            succ=`echo "$out" | tail -n 1 | cut --d " " -f 2`
            rate=`echo "$out" | head -n 1 | cut --d " " -f 3`

            if [[ "$succ" = "success." ]]; then
                successes=`echo $successes + 0.05 | bc -l`;
            fi

            rates=`echo "$rates $rate"`
            avg_rate=`echo "$avg_rate + $rate/20.0" | bc -l`
            echo "$step $m_factor $d_factor $successes $avg_rate $rates"
        done
    done
done
