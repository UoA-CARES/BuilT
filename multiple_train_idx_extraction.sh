#!/bin/bash

for i in {1..5}
do
    ./train_corrected.sh
done


for i in {1..5}
do
    ./train.sh
done




