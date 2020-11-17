#!/bin/bash

DEVICE_ID=0
CONFIG=tweet/tweet.yaml

[ ! -d "tweet/input" ] && cd tweet && sh download_data.sh && cd ..

export TOKENIZERS_PARALLELISM=true

CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CONFIG -f