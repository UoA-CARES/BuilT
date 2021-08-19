[![Build Status](https://travis-ci.com/UoA-CARES/BuilT.svg?branch=master)](https://travis-ci.com/UoA-CARES/BuilT)
[![codecov](https://codecov.io/gh/UoA-CARES/BuilT/branch/master/graph/badge.svg)](https://codecov.io/gh/UoA-CARES/BuilT)

# BuilT(Build a Trainer)
Easily build a trainer for your Depp Neural Network model and experiment as many as you want to find optimal combination of components(model, optimizer, scheduler) and hyper-parameters in a well-organized manner. 
- No more boilerplate code to train and evaluate your DNN model. just focus on your model. 
- Simply swap your dataset, model, optimizer and scheduler in the configuration file to find optimal combination. Your code doesn't need to be changed!!!. 
- Support Cross Validation, OOF(Out of Fold) Prediction 
- Support WandB(https://wandb.ai/) or tensorboard logging.
- Support checkpoint management(Save and load a model. Resume the previous training)
- BuilT easily integrates with Kaggle(https://www.kaggle.com/) notebook. (todo: add notebook link)

## Installation
Please follow the instruction below to install BuilT. 

### Installation of BuilT package from the source code
```
git clone https://github.com/UoA-CARES/BuilT.git
cd BuilT
python setup.py install
```

### Installation of BuilT package using pip
BuilT can be installed using pip(https://pypi.org/project/BuilT/). 
```
pip install built
```

## Usage

### Configuration
### Builder

### Trainer
### Dataset

### Model

### Loss
### Optimizer

### Scheduler

### Logger

### Metric

### Inference

### Ensemble



## Examples
### MNIST hand-written image classification
(todo)

### Sentiment Classification
(todo)


## Developer Guide
(todo)
```
conda create -n conda_BuilT python=3.7
conda activate conda_BuilT
pip install -r requirements.txt
```

# Reference
https://packaging.python.org/tutorials/packaging-projects/
