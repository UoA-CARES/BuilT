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

from torch.nn import functional as F
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tweet.src.dataset.tweet_dataset_base import TweetDatasetBase

@Registry.register(category='dataset')
class TweetDataset(TweetDatasetBase):
    def __init__(self, model_path, csv_path, transformer_type='roberta', train=False, split='train', max_len=96):
        super().__init__(model_path, csv_path, transformer_type, train, split, max_len)

    def encode_ids(self, encoding, row=None):
        ids = None
        if 'roberta' in self.transformer_type:
            ids = [0] + encoding.ids + [2]
        elif 'bert' in self.transformer_type:
            ids = [101] + encoding.ids[1:-1] + [102]
        else:
            raise RuntimeError(f'{self.transformer_type} is not supported')
        
        return ids

    def encode_offsets(self, encoding):
        offsets = [(0, 0)] + encoding.offsets + [(0, 0)]
        return offsets
