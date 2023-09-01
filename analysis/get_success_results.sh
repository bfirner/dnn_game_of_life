#! /bin/bash

# The paper plotted success vs m_factor and d_factor for different depths

echo "Steps M D SuccessRate AvgSuccess"

#for step in `seq 1 3`; do
#    for m_factor in 1 5 `seq 10 10 40`; do
#        for d_factor in 1 1.5 2; do
#            successes=0
#            rates=""
#            avg_rate=0
#            for trial in `seq 20`; do
#                out=`python3 ../gameoflife.py --step $step --m_factor $m_factor\
#                    --d_factor $d_factor  --use_sigmoid  --activation_fun LeakyReLU --normalize  \
#                    --batches 10000 --batch_size 128 --seed $trial --use_cuda \
#                    | grep -P "(Success rate|Training (success|failure))"`
#                succ=`echo "$out" | grep "Training" | cut --d " " -f 2`
#                rate=`echo "$out" | grep "Success rate" | cut --d " " -f 3`
#
#                if [[ "$succ" = "success." ]]; then
#                    successes=`echo $successes + 0.05 | bc -l`;
#                fi
#
#                rates=`echo "$rates $rate"`
#                avg_rate=`echo "$avg_rate + $rate/20.0" | bc -l`
#            done;
#            echo "$step $m_factor $d_factor $successes $avg_rate $rates"
#        done
#    done
#done

for step in 1 2 3 5 10; do
    for m_factor in 1 5 `seq 10 10 40`; do
        for d_factor in 1; do
            successes=0
            rates=""
            avg_rate=0
            for trial in `seq 20`; do
                out=`python3 ../gameoflife.py --step $step --m_factor $m_factor\
                    --d_factor $d_factor  --use_sigmoid  --activation_fun LeakyReLU  \
                    --batches 10000 --batch_size 128 --seed $trial --use_cuda \
                    | grep -P "(Success rate|Training (success|failure))"`
                succ=`echo "$out" | grep "Training" | cut --d " " -f 2`
                rate=`echo "$out" | grep "Success rate" | cut --d " " -f 3`

                if [[ "$succ" = "success." ]]; then
                    successes=`echo $successes + 0.05 | bc -l`;
                fi

                rates=`echo "$rates $rate"`
                avg_rate=`echo "$avg_rate + $rate/20.0" | bc -l`
            done;
            echo "$step $m_factor $d_factor $successes $avg_rate $rates"
        done
    done
done

for step in 2 3 5 10; do
    for m_factor in `seq 50 25 100`; do
        for d_factor in 1; do
            successes=0
            rates=""
            avg_rate=0
            for trial in `seq 20`; do
                out=`python3 ../gameoflife.py --step $step --m_factor $m_factor\
                    --d_factor $d_factor  --use_sigmoid  --activation_fun LeakyReLU  \
                    --batches 10000 --batch_size 128 --seed $trial --use_cuda \
                    | grep -P "(Success rate|Training (success|failure))"`
                succ=`echo "$out" | grep "Training" | cut --d " " -f 2`
                rate=`echo "$out" | grep "Success rate" | cut --d " " -f 3`

                if [[ "$succ" = "success." ]]; then
                    successes=`echo $successes + 0.05 | bc -l`;
                fi

                rates=`echo "$rates $rate"`
                avg_rate=`echo "$avg_rate + $rate/20.0" | bc -l`
            done;
            echo "$step $m_factor $d_factor $successes $avg_rate $rates"
        done
    done
done

for step in 5 10; do
    for m_factor in 1 5 `seq 125 25 150`; do
        for d_factor in 1; do
            successes=0
            rates=""
            avg_rate=0
            for trial in `seq 20`; do
                out=`python3 ../gameoflife.py --step $step --m_factor $m_factor\
                    --d_factor $d_factor  --use_sigmoid  --activation_fun LeakyReLU  \
                    --batches 10000 --batch_size 128 --seed $trial --use_cuda \
                    | grep -P "(Success rate|Training (success|failure))"`
                succ=`echo "$out" | grep "Training" | cut --d " " -f 2`
                rate=`echo "$out" | grep "Success rate" | cut --d " " -f 3`

                if [[ "$succ" = "success." ]]; then
                    successes=`echo $successes + 0.05 | bc -l`;
                fi

                rates=`echo "$rates $rate"`
                avg_rate=`echo "$avg_rate + $rate/20.0" | bc -l`
            done;
            echo "$step $m_factor $d_factor $successes $avg_rate $rates"
        done
    done
done
