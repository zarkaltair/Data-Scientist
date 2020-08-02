#!/usr/bin/gnuplot -persist

out_name = "plot.png"
in_name = "data2.csv"
column_1 = 2
column_2 = 3

set terminal png enhanced
set output "plot.png"
set size ratio 0.5
set title "Graph to data from file ".in_name." col ".column_1.", ".column_2
set xlabel "Time to process, s"
set ylabel "Something yet, %"
set xtics ("0" 0, "15" 15, "30" 30, "60" 60, "120" 120, "180" 180)
set autoscale y
set grid
set key autotitle columnhead

plot in_name using 1:column_1 smooth csplines notitle with lines lt 1,\
in_name using 1:column_1 with points poinsize 1 pointtype 7 lt 1,\
in_name using 1:column_2 smooth csplines notitle with lines lt 2,\
in_name using 1:column_2 with points pointsize 1 pointtype 13 lt 2
