#!/bin/bash

#set -x

function get_minimum {
    min=$1
        for i in $@; do
            if [ 1 -eq "$(echo "$min > $i" | bc)" ]; then
                min=$i
            fi
        done
    echo $min
}

function run_spmv {

    matrix_type=$1
    M_PATH="/mnt/data/matrices"
    M_LIST="matrices_names"
    if [ $matrix_type == "gen" ]; then
        M_PATH="/home/kupkupa/generated_matrices"
        M_LIST="generated_matrices_names"
    fi

    I_TEST=$2
    LOG_FILE=$3

    echo $matrix_type >> $LOG_FILE

    for matrix_name in `cat ~/${M_LIST}`; do

        time1=$(./spmv ${M_PATH}/${matrix_name}.crs $I_TEST | grep 'time' | cut -d ' ' -f3)
        time2=$(./spmv ${M_PATH}/${matrix_name}.crs $I_TEST | grep 'time' | cut -d ' ' -f3)
        time3=$(./spmv ${M_PATH}/${matrix_name}.crs $I_TEST | grep 'time' | cut -d ' ' -f3)
        time4=$(./spmv ${M_PATH}/${matrix_name}.crs $I_TEST | grep 'time' | cut -d ' ' -f3)
        time5=$(./spmv ${M_PATH}/${matrix_name}.crs $I_TEST | grep 'time' | cut -d ' ' -f3)

        min=$(get_minimum $time1 $time2 $time3 $time4 $time5)
        echo $min >> $LOG_FILE
    done
}

function run_spmm {

    matrix_type=$1
    M_PATH="/mnt/data/matrices"
    M_LIST="matrices_names"
    if [ $matrix_type == "gen" ]; then
        M_PATH="/home/kupkupa/generated_matrices"
        M_LIST="generated_matrices_names"
    fi

    I_TEST=$2
    NV=$3
    LOG_FILE=$4

    echo $matrix_type >> $LOG_FILE

    for matrix_name in `cat ~/${M_LIST}`; do

        time1=$(./spmm ${M_PATH}/${matrix_name}.crs $I_TEST $NV | grep 'time' | cut -d ' ' -f3)
        time2=$(./spmm ${M_PATH}/${matrix_name}.crs $I_TEST $NV | grep 'time' | cut -d ' ' -f3)
        time3=$(./spmm ${M_PATH}/${matrix_name}.crs $I_TEST $NV | grep 'time' | cut -d ' ' -f3)
        time4=$(./spmm ${M_PATH}/${matrix_name}.crs $I_TEST $NV | grep 'time' | cut -d ' ' -f3)
        time5=$(./spmm ${M_PATH}/${matrix_name}.crs $I_TEST $NV | grep 'time' | cut -d ' ' -f3)

        min=$(get_minimum $time1 $time2 $time3 $time4 $time5)
        echo $min >> $LOG_FILE
    done
}

###################################################################################################
if [ 0 -eq 1 ]; then
    for i in ALG0 ALG1 ALG2; do
        make ALG_TYPE=-D${i} FP_TYPE=-DFP64 OP_TYPE=-DAX_Y

        run_spmv "ssmc" 100 log.cusparse.ax_y.nv1.fp64.$i
        run_spmv "gen" 100 log.cusparse.ax_y.nv1.fp64.$i

        make ALG_TYPE=-D${i} FP_TYPE=-DFP32 OP_TYPE=-DAX_Y

        run_spmv "ssmc" 100 log.cusparse.ax_y.nv1.fp32.$i
        run_spmv "gen" 100 log.cusparse.ax_y.nv1.fp32.$i
    done
fi
###################################################################################################
if [ 0 -eq 1 ]; then
    for ORDER in ORDER_ROW ORDER_COL; do
        for i in ALG0 ALG1 ALG2 ALG3; do
            make ALG_TYPE=-D${i} ORDER_TYPE=-D${ORDER} FP_TYPE=-DFP64 OP_TYPE=-DAX_Y

            run_spmm "ssmc" 20 16 log.cusparse.ax_y.nv16.fp64.$ORDER.$i
            run_spmm "gen" 20 16 log.cusparse.ax_y.nv16.fp64.$ORDER.$i

            make ALG_TYPE=-D${i} ORDER_TYPE=-D${ORDER} FP_TYPE=-DFP32 OP_TYPE=-DAX_Y

            run_spmm "ssmc" 20 16 log.cusparse.ax_y.nv16.fp32.$ORDER.$i
            run_spmm "gen" 20 16 log.cusparse.ax_y.nv16.fp32.$ORDER.$i
        done
    done
fi
