#!/bin/bash

export DownloadPath=input
export RobertaBase=roberta-base

kaggle datasets download abhishek/$RobertaBase -p $DownloadPath/$RobertaBase --unzip
