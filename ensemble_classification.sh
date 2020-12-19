#!/bin/bash

DEVICE_ID=0
CONFIG=tweet/config/tweet_sentiment_classification.yaml


export TOKENIZERS_PARALLELISM=true

CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_sentiment with $CONFIG -f use_date=False
