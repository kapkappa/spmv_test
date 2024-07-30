#!/bin/bash

function get_minimum {
min=$1
for i in $@; do
    if [ 1 -eq "$(echo "$min > $i" | bc)" ]
    then
        min=$i
    fi
done
echo $min
}

make

for matrix_name in `cat ~/matrices_names`; do

    time1=$(./spmv /mnt/data/matrices/${matrix_name}.crs 100 | grep 'time' | cut -d ' ' -f3)
    time2=$(./spmv /mnt/data/matrices/${matrix_name}.crs 100 | grep 'time' | cut -d ' ' -f3)
    time3=$(./spmv /mnt/data/matrices/${matrix_name}.crs 100 | grep 'time' | cut -d ' ' -f3)
    time4=$(./spmv /mnt/data/matrices/${matrix_name}.crs 100 | grep 'time' | cut -d ' ' -f3)
    time5=$(./spmv /mnt/data/matrices/${matrix_name}.crs 100 | grep 'time' | cut -d ' ' -f3)

    min=$(get_minimum $time1 $time2 $time3 $time4 $time5)
    echo $min >> log.cusparse.ax_y.nv1.fp64
done

for matrix_name in `cat ~/matrices_names`; do

    time1=$(./spmm /mnt/data/matrices/${matrix_name}.crs 50 16 | grep 'time' | cut -d ' ' -f3)
    time2=$(./spmm /mnt/data/matrices/${matrix_name}.crs 50 16 | grep 'time' | cut -d ' ' -f3)
    time3=$(./spmm /mnt/data/matrices/${matrix_name}.crs 50 16 | grep 'time' | cut -d ' ' -f3)

    min=$(get_minimum $time1 $time2 $time3)
    echo $min >> log.cusparse.ax_y.nv16.fp64
done

make FP_TYPE=-DFP32

for matrix_name in `cat ~/matrices_names`; do

    time1=$(./spmv /mnt/data/matrices/${matrix_name}.crs 100 | grep 'time' | cut -d ' ' -f3)
    time2=$(./spmv /mnt/data/matrices/${matrix_name}.crs 100 | grep 'time' | cut -d ' ' -f3)
    time3=$(./spmv /mnt/data/matrices/${matrix_name}.crs 100 | grep 'time' | cut -d ' ' -f3)
    time4=$(./spmv /mnt/data/matrices/${matrix_name}.crs 100 | grep 'time' | cut -d ' ' -f3)
    time5=$(./spmv /mnt/data/matrices/${matrix_name}.crs 100 | grep 'time' | cut -d ' ' -f3)

    min=$(get_minimum $time1 $time2 $time3 $time4 $time5)
    echo $min >> log.cusparse.ax_y.nv1.fp32
done

for matrix_name in `cat ~/matrices_names`; do

    time1=$(./spmm /mnt/data/matrices/${matrix_name}.crs 50 16 | grep 'time' | cut -d ' ' -f3)
    time2=$(./spmm /mnt/data/matrices/${matrix_name}.crs 50 16 | grep 'time' | cut -d ' ' -f3)
    time3=$(./spmm /mnt/data/matrices/${matrix_name}.crs 50 16 | grep 'time' | cut -d ' ' -f3)

    min=$(get_minimum $time1 $time2 $time3)
    echo $min >> log.cusparse.ax_y.nv16.fp32
done
