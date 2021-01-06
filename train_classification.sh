#!/bin/bash

DEVICE_ID=0
CLASSIFICATION_CONFIG=tweet/config/tweet_sentiment_classification.yaml
INDEX_EXTRACTION_CONFIG=tweet/config/tweet_index_extraction.yaml

[ ! -d "tweet/input" ] && cd tweet && sh download_data.sh && cd ..
[ ! -d "tweet/input/roberta-base" ] && cd tweet && sh download_model_roberta.sh && cd ..

export TOKENIZERS_PARALLELISM=true

# CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CLASSIFICATION_CONFIG -f

CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CLASSIFICATION_CONFIG -f use_date=True
