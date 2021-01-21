#!/bin/bash

DEVICE_ID=0
DATASET=("TR" "TR_CORR")
TASK=("SC" "SE")
EXT=("En" "Es" "Esc")
TRANSFORMER=("BERT" "ROB" "ROB_L")
MODEL_CNT=1
# TRAIN_MODEL_NUMBERS=(20 23)
TRAIN_MODEL_NUMBERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24)




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
                        if [[ " ${TRAIN_MODEL_NUMBERS[@]} " =~ " ${MODEL_CNT} " ]]; then                            
                            CONFIG=${CLASSIFICATION_CONFIG_BASE}/$MODEL_CNT.[${DATASET[i]}]_[${TASK[j]}]_[${EXT[k]}]_[${TRANSFORMER[l]}].yaml
                            echo "========================================="
                            echo $CONFIG
                            echo "========================================="
                            
                            CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CONFIG -f use_date=False    
                        fi
                        #echo $MODEL_CNT
                        ((MODEL_CNT=MODEL_CNT+1))

                    done
                done
            else
                for m in {0..2} #transformer
                do
                    #echo $MODEL_CNT
                    if [[ " ${TRAIN_MODEL_NUMBERS[@]} " =~ " ${MODEL_CNT} " ]]; then                            
                        CONFIG=${CLASSIFICATION_CONFIG_BASE}/$MODEL_CNT.[${DATASET[i]}]_[${TASK[j]}]_[${TRANSFORMER[m]}].yaml
                        echo "========================================="
                        echo $CONFIG
                        echo "========================================="
                        
                        CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CONFIG -f use_date=False
                    fi
                    ((MODEL_CNT=MODEL_CNT+1))
                done
            fi
        done
    done
    MODEL_CNT=1
done


#echo $CONFIG

# [ ! -d "tweet/input" ] && cd tweet && sh download_data.sh && cd ..
# [ ! -d "tweet/input/bert-base-uncased" ] && cd tweet && sh download_model_bert.sh && cd ..


# export TOKENIZERS_PARALLELISM=true

# # CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CLASSIFICATION_CONFIG -f

# CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CLASSIFICATION_CONFIG -f use_date=True
