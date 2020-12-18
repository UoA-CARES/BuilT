#!/bin/bash

DEVICE_ID=0

CONFIG=tweet/config/tweet_index_extraction_coverage_correct.yaml

export TOKENIZERS_PARALLELISM=true

CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_SE_Esc with $CONFIG -f use_date=False
