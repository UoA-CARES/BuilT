#!/bin/bash

DEVICE_ID=0
CLASSIFICATION_CONFIG=tweet/config/tweet_sentiment_classification.yaml
INDEX_EXTRACTION_CONFIG=tweet/config/tweet_index_extraction_correct.yaml
COVERAGE_INDEX_EXTRACTION_CONFIG=tweet/config/tweet_index_extraction_coverage_correct.yaml

[ ! -d "tweet/input" ] && cd tweet && sh download_data.sh && cd ..

export TOKENIZERS_PARALLELISM=true

# CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CLASSIFICATION_CONFIG -f

CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $COVERAGE_INDEX_EXTRACTION_CONFIG -f use_date=True
