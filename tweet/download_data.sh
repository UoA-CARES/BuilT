#!/bin/bash

export KaggleCompName=tweet-sentiment-extraction
export DownloadPath=input
export RobertaBase=roberta-base

kaggle competitions download -c $KaggleCompName -p $DownloadPath

unzip $DownloadPath/$KaggleCompName.zip -d $DownloadPath/$KaggleCompName

kaggle datasets download abhishek/$RobertaBase -p $DownloadPath/$RobertaBase --unzip
