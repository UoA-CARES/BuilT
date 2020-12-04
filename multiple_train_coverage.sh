#!/bin/bash

for i in {1..4}
do
    ./train_coverage_corrected.sh
done

for i in {1..4}
do
    ./train_coverage.sh
done



