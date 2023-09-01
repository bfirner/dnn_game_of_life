set terminal pngcairo enhanced

set key outside top center


set key outside top center horizontal
d_factors="1 1.5 2"

set logscale y
set xlabel "Distance from known solution to minimum from training"
set ylabel "Loss"
set yrange [0.0001:]
# Some data ranges go to 31, but let's keep them all the same.
#set xrange [0:30]
#
#do for [step=2:3] {
#    do for [m_factor=10:40:10] {
#        do for [d_factor in d_factors] {
#            set output "straight_walks_results_".step."_steps".m_factor."_".d_factor.".png"
#            set title "Step ".step.", m factor ".m_factor.", d factor ".d_factor
#            plot for [trial=1:10] \
#                "straight_walk_results/error_walk_".step."_".m_factor."_".d_factor."_".trial.".txt" \
#                u 2:4 w lp notitle
#                #"Steps ".step.", m=".m_factor.", d=".d_factor.", trial ".trial
#        }
#    }
#    m_factor=1
#    d_factor=1
#    set output "straight_walks_results_".step."_steps".m_factor."_".d_factor.".png"
#    set title "Step ".step.", m factor ".m_factor.", d factor ".d_factor
#    plot for [trial=1:10] \
#        "straight_walk_results/error_walk_".step."_".m_factor."_".d_factor."_".trial.".txt" \
#        u 2:4 w lp notitle
#        #"Steps ".step.", m=".m_factor.", d=".d_factor.", trial ".trial
#}


unset xrange
unset title
steps="1 2 3"
m_factors="1 5 10"
do for [step in steps] {
    do for [m_factor in m_factors] {
        set output "bad_to_good_walk_step".step."_mfactor".m_factor.".png"
        #set title step." steps, m\\\_factor ".mfactor.", local to global minima."

        array max_values[10]
        do for [seed=1:10] {
            stats "bad_to_good_walk_results/error_walk_steps".step."_m".m_factor."_d1.0_seed".seed.".txt" using 2
            max_values[seed] = STATS_max
        }

        plot for [seed=1:10] \
            "bad_to_good_walk_results/error_walk_steps".step."_m".m_factor."_d1.0_seed".seed.".txt" \
            using (abs(max_values[seed] - $2)):4 w lp notitle
    }
}

unset xrange
unset title
steps="2 3"
m_factors="50"
do for [step in steps] {
    do for [m_factor in m_factors] {
        set output "bad_to_good_walk_step".step."_mfactor".m_factor.".png"
        #set title step." steps, m\\\_factor ".mfactor.", local to global minima."

        array max_values[10]
        do for [seed=1:10] {
            stats "bad_to_good_walk_results/error_walk_steps".step."_m".m_factor."_d1.0_seed".seed.".txt" using 2
            max_values[seed] = STATS_max
        }

        plot for [seed=1:10] \
            "bad_to_good_walk_results/error_walk_steps".step."_m".m_factor."_d1.0_seed".seed.".txt" \
            using (abs(max_values[seed] - $2)):4 w lp notitle
    }
}

unset xrange
unset title
steps="5 10 20"
m_factors="1 5 50 100"
do for [step in steps] {
    do for [m_factor in m_factors] {
        set output "bad_to_good_walk_step".step."_mfactor".m_factor.".png"
        #set title step." steps, m\\\_factor ".mfactor.", local to global minima."

        array max_values[10]
        do for [seed=1:10] {
            stats "bad_to_good_walk_results/error_walk_steps".step."_m".m_factor."_d1.0_seed".seed.".txt" using 2
            max_values[seed] = STATS_max
        }

        plot for [seed=1:10] \
            "bad_to_good_walk_results/error_walk_steps".step."_m".m_factor."_d1.0_seed".seed.".txt" \
            using (abs(max_values[seed] - $2)):4 w lp notitle
    }
}
