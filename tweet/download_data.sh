#!/bin/bash

export KaggleCompName=tweet-sentiment-extraction
export DownloadPath=input
export RobertaBase=roberta-base
export BertBaseUncased=bert-base-uncased

kaggle competitions download -c $KaggleCompName -p $DownloadPath

unzip $DownloadPath/$KaggleCompName.zip -d $DownloadPath/$KaggleCompName

kaggle datasets download abhishek/$RobertaBase -p $DownloadPath/$RobertaBase --unzip
kaggle datasets download abhishek/$BertBaseUncased -p $DownloadPath/$BertBaseUncased --unzip
