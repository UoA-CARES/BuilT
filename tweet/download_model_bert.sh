#!/bin/bash

export DownloadPath=input
export BertBaseUncased=bert-base-uncased

kaggle datasets download abhishek/$BertBaseUncased -p $DownloadPath/$BertBaseUncased --unzip
