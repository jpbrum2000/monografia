#!/bin/bash
for i in {1..1050}; do
    if (( $i % 2 != 0 ))
    then
        ./knn_final.py $i > result_knn_$i &
    fi
    
done
