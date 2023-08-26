set terminal pngcairo enhanced

set key outside top center

set ylabel "Success Rate (out of 20 trials)"
set xlabel "Multiple of Minimum Feature Maps"
set yrange [0:1]
set output "2_step_gol_results_line.png"
plot for [depth in  "1 1.5 2"] "<grep -P '^2 [01234]* ".depth." ' 2022-04-01-success_results.dat" u 2:4 w lp title "Depth ".depth
set output "3_step_gol_results_line.png"
plot for [depth in  "1 1.5 2"] "<grep -P '^3 [01234]* ".depth." ' 2022-04-01-success_results.dat" u 2:4 w lp title "Depth ".depth

set ylabel "Average Ratio of Correct Predictions"
set output "2_step_gol_results_line_avg.png"
plot for [depth in  "1 1.5 2"] "<grep -P '^2 [01234]* ".depth." ' 2022-04-01-success_results.dat" u 2:5 w lp title "Depth ".depth
set output "3_step_gol_results_line_avg.png"
plot for [depth in  "1 1.5 2"] "<grep -P '^3 [01234]* ".depth." ' 2022-04-01-success_results.dat" u 2:5 w lp title "Depth ".depth

unset yrange

set output "2_step_gol_results.png"

set xlabel "Multiple of Minimum Feature Maps" rotate parallel
set ylabel "Depth Multiple" rotate parallel
set zlabel "Success Rate (out of 20 trials)" rotate parallel


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

splot "<grep '^2' 2022-04-01-success_results.dat" u 2:3:4:(rgb(128+$2*20, 128 + 100*$3, 128+128*$4)) with boxes fc rgb variable notitle


set output "3_step_gol_results.png"

splot "<grep '^3' 2022-04-01-success_results.dat" u 2:3:4:(rgb(128+$2*20, 128 + 100*$3, 128+128*$4)) with boxes fc rgb variable notitle
