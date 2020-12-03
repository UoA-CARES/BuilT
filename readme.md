[![Build Status](https://travis-ci.com/UoA-CARES/BuilT.svg?branch=master)](https://travis-ci.com/UoA-CARES/BuilT)
[![codecov](https://codecov.io/gh/UoA-CARES/BuilT/branch/master/graph/badge.svg)](https://codecov.io/gh/UoA-CARES/BuilT)

# BuilT(Build a Trainer)
Easily build a trainer for DNNs using GUI and quickly do experiments for them. 

## Installation
```bash
conda create -n conda_BuilT python=3.7
conda activate conda_BuilT
git clone https://github.com/UoA-CARES/BuilT.git
cd BuilT
git checkout nlp
pip install -r requirements-gpu.txt
```

## Kaggle API credentials

To use the Kaggle API, sign up for a Kaggle account at https://www.kaggle.com. Then go to the 'Account' tab of your user profile (`https://www.kaggle.com/<username>/account`) and select 'Create API Token'. This will trigger the download of `kaggle.json`, a file containing your API credentials. Place this file in the location `~/.kaggle/kaggle.json` (on Windows in the location `C:\Users\<Windows-username>\.kaggle\kaggle.json` - you can check the exact location, sans drive, with `echo %HOMEPATH%`). You can define a shell environment variable `KAGGLE_CONFIG_DIR` to change this location to `$KAGGLE_CONFIG_DIR/kaggle.json` (on Windows it will be `%KAGGLE_CONFIG_DIR%\kaggle.json`).

For your security, ensure that other users of your computer do not have read access to your credentials. On Unix-based systems you can do this with the following command: 

`chmod 600 ~/.kaggle/kaggle.json`

You can also choose to export your Kaggle username and token to the environment:

```bash
export KAGGLE_USERNAME=datadinosaur
export KAGGLE_KEY=xxxxxxxxxxxxxx
```

## Training with BuilT
With BuilT, users can train task-oriented models such as classification, index extraction or ensenble with minimum changes and efforts.
The following two examples demonstrate Tweet sentiment classification and start-end index extraction tasks.

### Train Tweet Classification Model
This can be acheived by setting up 
```bash
conda activate conda_BuilT
cd BuilT
vi ./train.sh 
```
You can see there are two configuration files defined as `CLASSIFICATION_CONFIG` and `INDEX_EXTRACTION_CONFIG`. If you want to train Tweet sentiment classifier, then pass `$CLASSIFICATION_CONFIG` to `run.py` as follows:
```bash
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $CLASSIFICATION_CONFIG -f use_date=True
```
This command implies that we want to use `$DEVICE_ID` for `train` task with `$CLASSIFICATION_CONFIG` configuration. If `use_date` is set as True, logging information will be stored under `train_dirs/tweet_sentiment_classification/roberta-base/20201201-14138`. The forlder is formatted as `YYYYMMDD-HHMMSS` (Y:year, M: month, D:date).

Train will start with the following command
```bash
./train.sh
```


### Train Tweet Index extraction Model
To do this task, all you need is to pass `INDEX_EXTRACTION_CONFIG` to `run.py`. Your `train.sh` should look like
```bash
CUDA_VISIBLE_DEVICES=$DEVICE_ID python run.py train with $INDEX_EXTRACTION_CONFIG -f use_date=True
```
For sure you can create another script `train_index_extraction.sh` for later use (e.g., multiple runs or hyperparameters sweeping).

### Multiple runs of train script
If you want to run 4 multiple times with varying random seed or configuration files, this can be also easily acheived with BuilT.
We create a shell script with name `multiple_train.sh` as follows (assuming that you already created `train_index_extraction.sh` from above step).

```bash
#!/bin/bash
for i in {1..4}
do
    ./train.sh
done

for i in {1..4}
do
    ./train_index_extraction.sh
done
```
and run it 
```bash
./multiple_train.sh
```
This should perform 4 times for classifcation model and index extraction trainings.



## Tensorboard Visualisation
```bash
tensorboard --logdir train_dirs/tweet_classification
```

## wandb logging and visualisation
