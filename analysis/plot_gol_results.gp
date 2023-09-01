set terminal pngcairo enhanced size 480, 320 fontscale 1

set key outside top center horizontal

# The user can provide a filename with `gnuplot -e "datafile='file'" plot_gol_results.gp`
if (!exists("datafile")) datafile="2023-08-28-success_results.dat 2023-08-29-success_results.dat"

set linetype 1 lw 2 pt 5
set linetype 2 lw 2 pt 7
set linetype 3 lw 2 pt 9

set ylabel "Model Success Rate\n(over 20 trials)"
set xlabel "Multiple of Feature Maps (m\\\_factor)"
set yrange [0:1]
set output "1_step_gol_results_line.png"
plot for [depth in  "1 1.5 2"] "<grep -P '^1 [0-9]* ".depth." ' ".datafile u 2:4 w lp title "Depth (d\\\_factor) ".depth
set output "2_step_gol_results_line.png"
plot for [depth in  "1 1.5 2"] "<grep -P '^2 [0-9]* ".depth." ' ".datafile u 2:4 w lp title "Depth (d\\\_factor) ".depth
set output "3_step_gol_results_line.png"
plot for [depth in  "1 1.5 2"] "<grep -P '^3 [0-9]* ".depth." ' ".datafile u 2:4 w lp title "Depth (d\\\_factor) ".depth

set ylabel "Ratio of Correct Predictions\n(averaged over 20 trials)"
set output "1_step_gol_results_line_avg.png"
plot for [depth in  "1 1.5 2"] "<grep -P '^1 [0-9]* ".depth." ' ".datafile u 2:5 w lp title "Depth (d\\\_factor) ".depth
set output "2_step_gol_results_line_avg.png"
plot for [depth in  "1 1.5 2"] "<grep -P '^2 [0-9]* ".depth." ' ".datafile u 2:5 w lp title "Depth (d\\\_factor) ".depth
set output "3_step_gol_results_line_avg.png"
plot for [depth in  "1 1.5 2"] "<grep -P '^3 [0-9]* ".depth." ' ".datafile u 2:5 w lp title "Depth (d\\\_factor) ".depth


# Combined graphs with set depth, but only for 1-3 steps
set ylabel "Model Success Rate\n(over 20 trials)"
set xlabel "Multiple of Feature Maps (m\\\_factor)"
set output "1_depth_small_steps_gol_results_line.png"
plot for [step in  "1 2 3"] "<grep -P '^".step." [0-9]* 1 ' ".datafile u 2:4 w lp title step." steps"
set ylabel "Ratio of Correct Predictions\n(averaged over 20 trials)"
set output "1_depth_small_steps_gol_results_line_avg.png"
plot for [step in  "1 2 3"] "<grep -P '^".step." [0-9]* 1 ' ".datafile u 2:5 w lp title step." steps"


# Combined graphs with set depth
set ylabel "Model Success Rate\n(over 20 trials)"
set xlabel "Multiple of Feature Maps (m\\\_factor)"
set output "1_depth_gol_results_line.png"
plot for [step in  "1 2 3 5 10 20"] "<grep -P '^".step." [0-9]* 1 ' ".datafile u 2:4 w lp title step." steps"
set ylabel "Ratio of Correct Predictions\n(averaged over 20 trials)"
set output "1_depth_gol_results_line_avg.png"
plot for [step in  "1 2 3 5 10 20"] "<grep -P '^".step." [0-9]* 1 ' ".datafile u 2:5 w lp title step." steps"


unset yrange
unset xrange

set output "2_step_gol_results.png"

set xlabel "Multiple of Feature Maps (m\\\_factor)" rotate parallel
set ylabel "Depth Multiple (d\\\_factor)" rotate parallel
set zlabel "Success Rate\n(over 20 trials)" rotate parallel


set boxwidth 2 abs
set boxdepth 0.1
set xyplane at 0
set grid x z vertical lw 1.0
set view 59, 24
set pm3d border lc black lighting

set ytics 1,0.5,2
set xtics 0,10,40

# Data is in the following columns:
# Steps M D SuccessRate AvgSuccess

set style fill solid

rgb(r,g,b) = 65536 * int(r) + 256 * int(g) + int(b)

splot "<grep '^2' ".datafile u 2:3:4:(rgb(128+$2*20, 128 + 100*$3, 128+128*$4)) with boxes fc rgb variable notitle


set output "3_step_gol_results.png"

splot "<grep '^3' ".datafile u 2:3:4:(rgb(128+$2*20, 128 + 100*$3, 128+128*$4)) with boxes fc rgb variable notitle
