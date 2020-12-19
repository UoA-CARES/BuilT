#!/bin/bash

DEVICE_ID=0

CONFIG=tweet/config/tweet_index_extraction_correct.yaml

export TOKENIZERS_PARALLELISM=true

CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_index_extraction with $CONFIG -f use_date=False
