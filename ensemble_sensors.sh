#!/bin/bash

DEVICE_ID=0
DATASET=("TR" "TR_CORR")
TASK=("SC" "SE")
EXT=("En" "Es" "Esc")
TRANSFORMER=("BERT" "ROB" "ROB_L")
MODEL_CNT=1

CLASSIFICATION_CONFIG_BASE="tweet/config/for_sensors"
export TOKENIZERS_PARALLELISM=true

#1.[TR]_[SC]_[BERT].yaml
for o in {0..0}
do
    echo "==================================="
    echo $o run
    echo "==================================="
    for i in {0..1} #dataset
    do
        for j in {0..1} #task
        do
            if [ ${TASK[j]} == "SE" ]; then
                for k in {0..2} #Ext
                do
                    for l in {0..2} #transformer
                    do
                        #echo $MODEL_CNT
                        CONFIG=${CLASSIFICATION_CONFIG_BASE}/$MODEL_CNT.[${DATASET[i]}]_[${TASK[j]}]_[${EXT[k]}]_[${TRANSFORMER[l]}].yaml
                        echo "========================================="
                        echo $CONFIG
                        echo "========================================="
                        ((MODEL_CNT=MODEL_CNT+1))
                        # CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CONFIG -f use_date=False

                        SUFFIX="[${DATASET[i]}]_[${TASK[j]}]_[${EXT[k]}]_[${TRANSFORMER[l]}]"
                        if [ ${EXT[k]} == "En" ] || [ ${EXT[k]} == "Es" ]; then
                            CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_index_extraction with $CONFIG \
                                ensemble_train=False \
                                ensemble_split='test' \
                                ensemble_csv_input_path='tweet/input/tweet-sentiment-extraction/corrected_new_test.csv' \
                                ensemble_csv_output_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_$SUFFIX.csv' \
                                original_selected_text_column='selected_text' \
                                -f
                        elif [ ${EXT[k]} == "Esc" ]; then
                            CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_SE_Esc with $CONFIG \
                                ensemble_train=False \
                                ensemble_split='test' \
                                ensemble_csv_input_path='tweet/input/tweet-sentiment-extraction/corrected_new_test.csv' \
                                ensemble_csv_output_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_$SUFFIX.csv' \
                                original_selected_text_column='selected_text' \
                                -f
                        fi
                    done
                done
            else
                for m in {0..2} #transformer
                do
                    #echo $MODEL_CNT
                    CONFIG=${CLASSIFICATION_CONFIG_BASE}/$MODEL_CNT.[${DATASET[i]}]_[${TASK[j]}]_[${TRANSFORMER[m]}].yaml
                    echo "========================================="
                    echo $CONFIG
                    echo "========================================="
                    ((MODEL_CNT=MODEL_CNT+1))
                    # CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CONFIG -f use_date=False

                    SUFFIX="[${DATASET[i]}]_[${TASK[j]}]_[${EXT[k]}]_[${TRANSFORMER[l]}]"
                    
                    CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py ensemble_sentiment with $CONFIG \
                        ensemble_train=False \
                        ensemble_split='test' \
                        ensemble_csv_input_path='tweet/input/tweet-sentiment-extraction/corrected_new_test.csv' \
                        ensemble_csv_output_path='tweet/input/tweet-sentiment-extraction/corrected_new_test_$SUFFIX.csv' \
                        -f
                done
            fi
        done
    done
    MODEL_CNT=1
done
