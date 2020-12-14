#!/bin/bash

export KaggleCompName=tweet-sentiment-extraction
export DownloadPath=input
export RobertaBase=roberta-base
export RobertaLarge=roberta-large
export BertBaseUncased=bert-base-uncased

BERT=https://github.com/inkyusa/BuilT/releases/download/v0.1/bert-base-uncased.zip
ROBERTA_BASE=https://github.com/inkyusa/BuilT/releases/download/v0.1/roberta-base.zip
ROBERTA_LARGE=https://github.com/inkyusa/BuilT/releases/download/v0.1/roberta-large.zip

TRAIN=https://github.com/inkyusa/BuilT/releases/download/v0.1/train.csv
TRAIN_CORR=https://github.com/inkyusa/BuilT/releases/download/v0.1/train_corrected.csv
if [ ! -d "${DownloadPath}" ]; then
  echo "=================================="
  echo "   Creating ${DownloadPath} folder  "
  echo "=================================="
  mkdir ${DownloadPath}
fi

#=====================
# Roberta-base
#=====================
if [ ! -d "input/roberta-base" ]; then
  if [ ! -f "${DownloadPath}/${RobertaBase}.zip" ]; then
    echo "=================================="
    echo "  Downloading ${ROBERTA_BASE} to ${DownloadPath}"
    echo "=================================="
    wget ${ROBERTA_BASE} -P ${DownloadPath}
  fi
  echo "=================================="
  echo "  Uncompressing ${ROBERTA_BASE}"
  echo "=================================="
  unzip -d ${DownloadPath} "${DownloadPath}/${RobertaBase}.zip"
  echo "=================================="
  echo "  Delete ${RobertaBase}.zip"
  echo "=================================="
  rm "${DownloadPath}/${RobertaBase}.zip"
fi

#=====================
# Roberta-large
#=====================
if [ ! -d "input/roberta-large" ]; then
  if [ ! -f "${DownloadPath}/${RobertaLarge}.zip" ]; then
    echo "=================================="
    echo "  Downloading ${ROBERTA_LARGE} to ${DownloadPath}"
    echo "=================================="
    wget ${ROBERTA_LARGE} -P ${DownloadPath}
  fi
  echo "=================================="
  echo "  Uncompressing ${ROBERTA_LARGER}"
  echo "=================================="
  unzip -d ${DownloadPath} "${DownloadPath}/${RobertaLarge}.zip"
  echo "=================================="
  echo "  Delete ${RobertaLarge}.zip"
  echo "=================================="
  rm "${DownloadPath}/${RobertaLarge}.zip"
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
if [ ! -f "${DownloadPath}/${KaggleCompName}/new_train.csv" ]; then
  echo "=================================="
  echo "     Downloading new_train.csv        "
  echo "=================================="
  wget ${TRAIN} -P "${DownloadPath}/${KaggleCompName}"
fi

if [ ! -f "${DownloadPath}/${KaggleCompName}/corrected_new_train.csv" ]; then
  echo "=================================="
  echo " Downloading corrected_new_train.csv  "
  echo "=================================="
  wget ${TRAIN_CORR} -P "${DownloadPath}/${KaggleCompName}"
fi

if [ ! -f "${DownloadPath}/${KaggleCompName}/new_test.csv" ]; then
  echo "=================================="
  echo "     Downloading new_test.csv        "
  echo "=================================="
  wget ${TRAIN} -P "${DownloadPath}/${KaggleCompName}"
fi

if [ ! -f "${DownloadPath}/${KaggleCompName}/corrected_new_test.csv" ]; then
  echo "=================================="
  echo " Downloading corrected_new_test.csv  "
  echo "=================================="
  wget ${TRAIN_CORR} -P "${DownloadPath}/${KaggleCompName}"
fi