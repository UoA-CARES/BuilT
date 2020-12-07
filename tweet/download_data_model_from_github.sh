#!/bin/bash

export KaggleCompName=tweet-sentiment-extraction
export DownloadPath=input
export RobertaBase=roberta-base
export BertBaseUncased=bert-base-uncased

BERT=https://github.com/inkyusa/BuilT/releases/download/v0.1/bert-base-uncased.zip
ROBERTA=https://github.com/inkyusa/BuilT/releases/download/v0.1/roberta-base.zip
TRAIN=https://github.com/inkyusa/BuilT/releases/download/v0.1/train.csv
TRAIN_CORR=https://github.com/inkyusa/BuilT/releases/download/v0.1/train_corrected.csv
if [ ! -d "${DownloadPath}" ]; then
  echo "=================================="
  echo "   Creating ${DownloadPath} folder  "
  echo "=================================="
  mkdir ${DownloadPath}
fi

#=====================
# Roberta
#=====================
if [ ! -d "input/roberta-base" ]; then
  if [ ! -f "${DownloadPath}/${RobertaBase}.zip" ]; then
    echo "=================================="
    echo "  Downloading ${ROBERTA} to ${DownloadPath}"
    echo "=================================="
    wget ${ROBERTA} -P ${DownloadPath}
  fi
  echo "=================================="
  echo "  Uncompressing ${ROBERTA}"
  echo "=================================="
  unzip -d ${DownloadPath} "${DownloadPath}/${RobertaBase}.zip"
  echo "=================================="
  echo "  Delete ${RobertaBase}.zip"
  echo "=================================="
  rm "${DownloadPath}/${RobertaBase}.zip"
fi

#=====================
# Bert
#=====================

if [ ! -d "input/bert-base-uncased" ]; then
  if [ ! -f "${DownloadPath}/${BertBaseUncased}.zip" ]; then
    echo "=================================="
    echo "  Downloading ${BERT} to ${DownloadPath}"
    echo "=================================="
    wget ${BERT} -P ${DownloadPath}
  fi
  echo "=================================="
  echo "  Uncompressing ${BERT}"
  echo "=================================="
  unzip -d ${DownloadPath} "${DownloadPath}/${BertBaseUncased}.zip"
  echo "=================================="
  echo "  Delete ${BertBaseUncased}.zip"
  echo "=================================="
  rm "${DownloadPath}/${BertBaseUncased}.zip"
fi

#=====================
# Dataset
#=====================
if [ ! -f "${DownloadPath}/${KaggleCompName}/train.csv" ]; then
  echo "=================================="
  echo "     Downloading train.csv        "
  echo "=================================="
  wget ${TRAIN} -P "${DownloadPath}/${KaggleCompName}"
fi

if [ ! -f "${DownloadPath}/${KaggleCompName}/train_corrected.csv" ]; then
  echo "=================================="
  echo " Downloading train_corrected.csv  "
  echo "=================================="
  wget ${TRAIN_CORR} -P "${DownloadPath}/${KaggleCompName}"
fi