#!/bin/bash

DEVICE_ID=0
CONFIG=config/mnist.yaml

CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CONFIG -f