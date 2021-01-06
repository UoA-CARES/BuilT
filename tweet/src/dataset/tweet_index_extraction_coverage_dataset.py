from sklearn import model_selection
import numpy as np
import pandas as pd
import os
import tokenizers
import string
import torch
import re
import transformers
import torch.nn as nn

from built.registry import Registry
# from tweet.src.dataset.tweet_dataset import TweetDataset

from torch.nn import functional as F
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tweet.src.dataset.tweet_dataset_base import TweetDatasetBase


@Registry.register(category='dataset')
class TweetIndexExtractionCoverageDataset(TweetDatasetBase):
    def __init__(self, model_path, csv_path, transformer_type='roberta', train=False, split='train', max_len=96):
        super().__init__(model_path, csv_path, transformer_type, train, split, max_len)

    def encode_ids(self, encoding, row=None):
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        coverage = int(len(row.selected_text) / len(row.text) * 15.0)  # 0 ~ 10
        coverage_id = self.tokenizer.encode(str(coverage)).ids

        ids = None
        if 'roberta' in self.transformer_type:
            ids = [0] + sentiment_id + coverage_id + \
                [2, 2] + encoding.ids + [2]
        elif 'bert' in self.transformer_type:
            ids = [101] + sentiment_id[1:-1] + coverage_id[1:-1] + \
                [102] + encoding.ids[1:-1] + [102]
        else:
            raise RuntimeError(f'{self.transformer_type} is not supported')

        return ids

    def encode_offsets(self, encoding):
        offsets = None

        if 'roberta' in self.transformer_type:
            offsets = [(0, 0)] * 5 + encoding.offsets + [(0, 0)]
        elif 'bert' in self.transformer_type:
            offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]
        else:
            raise RuntimeError(f'{self.transformer_type} is not supported')
        return offsets
