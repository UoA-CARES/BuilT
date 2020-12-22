#!/bin/bash
export TOKENIZERS_PARALLELISM=true

DEVICE_ID=0

CONFIG=tweet/config/tweet_sentiment_classification.yaml
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_sentiment with $CONFIG \
  ensemble_train=False \
  ensemble_split='test' \
  ensemble_csv_input_path='tweet/input/tweet-sentiment-extraction/corrected_new_test.csv' \
  ensemble_csv_output_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment.csv' \
  -f


CONFIG=tweet/config/tweet_index_extraction_correct.yaml
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_index_extraction with $CONFIG \
  ensemble_train=False \
  ensemble_split='test' \
  ensemble_csv_input_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment.csv' \
  ensemble_csv_output_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index.csv' \
  original_selected_text_column='selected_text' \
  -f


CONFIG=tweet/config/tweet_index_extraction_coverage_correct.yaml
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_SE_Esc_using_Es with $CONFIG \
  ensemble_train=False \
  ensemble_split='test' \
  ensemble_csv_input_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index.csv' \
  ensemble_csv_output_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index_Coverage.csv' \
  original_selected_text_column='selected_text' \
  pred_selected_text_column='selected_text' \
  -f