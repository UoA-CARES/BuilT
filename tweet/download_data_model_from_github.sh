#!/bin/bash

export KaggleCompName=tweet-sentiment-extraction
export DownloadPath=input
export RobertaBase=roberta-base
export RobertaLarge=roberta-large
export BertBaseUncased=bert-base-uncased

https://github.com/UoA-CARES/BuilT/releases/download/v0.1/new_test.csv


BERT=https://github.com/UoA-CARES/BuilT/releases/download/v0.1/bert-base-uncased.zip
ROBERTA_BASE=https://github.com/UoA-CARES/BuilT/releases/download/v0.1/roberta-base.zip
ROBERTA_LARGE=https://github.com/UoA-CARES/BuilT/releases/download/v0.1/roberta-large.zip

TRAIN=https://github.com/UoA-CARES/BuilT/releases/download/v0.1/train.csv
NEW_TRAIN=https://github.com/UoA-CARES/BuilT/releases/download/v0.1/new_train.csv
NEW_TEST=https://github.com/UoA-CARES/BuilT/releases/download/v0.1/new_test.csv
CORR_NEW_TRAIN=https://github.com/UoA-CARES/BuilT/releases/download/v0.1/corrected_new_train.csv
CORR_NEW_TEST=https://github.com/UoA-CARES/BuilT/releases/download/v0.1/corrected_new_test.csv

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

if [ ! -f "${DownloadPath}/${KaggleCompName}/train.csv" ]; then
  echo "=================================="
  echo "     Downloading train.csv        "
  echo "=================================="
  wget ${TRAIN} -P "${DownloadPath}/${KaggleCompName}"
fi

if [ ! -f "${DownloadPath}/${KaggleCompName}/new_train.csv" ]; then
  echo "=================================="
  echo "     Downloading new_train.csv        "
  echo "=================================="
  wget ${NEW_TRAIN} -P "${DownloadPath}/${KaggleCompName}"
fi

if [ ! -f "${DownloadPath}/${KaggleCompName}/corrected_new_train.csv" ]; then
  echo "=================================="
  echo " Downloading corrected_new_train.csv  "
  echo "=================================="
  wget ${CORR_NEW_TRAIN} -P "${DownloadPath}/${KaggleCompName}"
fi

if [ ! -f "${DownloadPath}/${KaggleCompName}/new_test.csv" ]; then
  echo "=================================="
  echo "     Downloading new_test.csv        "
  echo "=================================="
  wget ${NEW_TEST} -P "${DownloadPath}/${KaggleCompName}"
fi

if [ ! -f "${DownloadPath}/${KaggleCompName}/corrected_new_test.csv" ]; then
  echo "=================================="
  echo " Downloading corrected_new_test.csv  "
  echo "=================================="
  wget ${CORR_NEW_TEST} -P "${DownloadPath}/${KaggleCompName}"
fi