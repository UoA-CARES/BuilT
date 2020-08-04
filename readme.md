[![Build Status](https://travis-ci.com/UoA-CARES/BuilT.svg?branch=master)](https://travis-ci.com/UoA-CARES/BuilT)
[![codecov](https://codecov.io/gh/UoA-CARES/BuilT/branch/master/graph/badge.svg)](https://codecov.io/gh/UoA-CARES/BuilT)

# BuilT(Build Trainer)
Easily build a trainer for DNNs using GUI and quickly do experiments for them. 

## Installation
```
conda create -n conda_BuilT python=3.7
conda activate conda_BuilT
pip install requirements-gpu.txt
```

## How to run MNIST example
```
conda activate conda_BuilT
git clone https://github.com/UoA-CARES/BuilT.git
cd BuilT
sh ./train.sh
```
