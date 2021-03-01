#!/bin/bash
export TOKENIZERS_PARALLELISM=true

DEVICE_ID=0

CONFIG=tweet/config/tweet_sentiment_classification.yaml
CONFIG=tweet/config/for_sensors/14.[TR_CORR]_[SC]_[ROB].yaml
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_sentiment with $CONFIG \
 ensemble_train=False \
 ensemble_split='test' \
 ensemble_csv_input_path='tweet/input/tweet-sentiment-extraction/corrected_new_test.csv' \
 ensemble_csv_output_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_output.csv' \
 -f


CONFIG=tweet/config/for_sensors/20.[TR_CORR]_[SE]_[Es]_[ROB].yaml
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_index_extraction with $CONFIG \
  ensemble_train=False \
  ensemble_split='test' \
  ensemble_csv_input_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_output.csv' \
  ensemble_csv_output_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index.csv' \
  original_selected_text_column='selected_text' \
  -f

CONFIG=tweet/config/for_sensors/23.[TR_CORR]_[SE]_[Esc]_[ROB].yaml
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_SE_Esc_using_Es with $CONFIG \
  ensemble_train=False \
  ensemble_split='test' \
  ensemble_csv_input_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index.csv' \
  ensemble_csv_output_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index_Coverage.csv' \
  original_selected_text_column='ori_selected_text' \
  pred_selected_text_column='selected_text' \
  -f