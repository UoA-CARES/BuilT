#!/bin/bash

DEVICE_ID=0
CONFIG=tweet/tweet.yaml

CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CONFIG -f