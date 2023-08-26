set terminal pngcairo enhanced

set key outside top center


set key outside top center horizontal
d_factors="1 1.5 2"

set logscale y
set xlabel "Distance from local minima to global\nalong shortest path"
set ylabel "Loss"
# Some data ranges go to 31, but let's keep them all the same.
set xrange [0:30]

do for [step=2:3] {
    do for [m_factor=10:40:10] {
        do for [d_factor in d_factors] {
            set output "straight_walks_results_".step."_steps".m_factor."_".d_factor.".png"
            set title "Step ".step.", m factor ".m_factor.", d factor ".d_factor
            plot for [trial=1:10] \
                "straight_walk_results/error_walk_".step."_".m_factor."_".d_factor."_".trial.".txt" \
                u 2:4 w lp notitle
                #"Steps ".step.", m=".m_factor.", d=".d_factor.", trial ".trial
        }
    }
    m_factor=1
    d_factor=1
    set output "straight_walks_results_".step."_steps".m_factor."_".d_factor.".png"
    set title "Step ".step.", m factor ".m_factor.", d factor ".d_factor
    plot for [trial=1:10] \
        "straight_walk_results/error_walk_".step."_".m_factor."_".d_factor."_".trial.".txt" \
        u 2:4 w lp notitle
        #"Steps ".step.", m=".m_factor.", d=".d_factor.", trial ".trial
}


unset xrange
m_factor=1
d_factor=1
set output "bad_to_good_walk.png"
set title "2 Steps, m factor 1, d factor 1, local to global minima."
plot for [trial=2:11] \
    "bad_to_good_walk_results/error_walk_seed_".trial.".txt" \
    u 2:4 w lp notitle
