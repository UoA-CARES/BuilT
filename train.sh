#!/bin/bash

DEVICE_ID=0
CLASSIFICATION_CONFIG=tweet/config/tweet_sentiment_classification.yaml

[ ! -d "tweet/input" ] && cd tweet && sh download_data.sh && cd ..

export TOKENIZERS_PARALLELISM=true

CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CLASSIFICATION_CONFIG -f