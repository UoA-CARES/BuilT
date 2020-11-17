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

## Train Tweet Classification Model
```bash
conda activate conda_BuilT
cd BuilT
sh ./train.sh
```

## Tensorboard Visualization
```bash
tensorboard --logdir train_dirs/tweet_classification
```
