#!/bin/bash

for i in {1..5}
do
    sh train_corrected.sh
done


for i in {1..5}
do
    sh train.sh
done




