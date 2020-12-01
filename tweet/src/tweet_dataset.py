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


@Registry.register(category='dataset')
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, model_path, csv_path, train=False, max_len=96):
        df = pd.read_csv(csv_path)
        self.df = df.dropna().reset_index(drop=True)
        self.max_len = max_len
        self.labeled = 'selected_text' in self.df
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab_file=model_path+'vocab.json', 
            merges_file=model_path+'merges.txt', 
            lowercase=True,
            add_prefix_space=True)

    def __getitem__(self, index):
        data = {}
        target = {}

        row = self.df.iloc[index]
        #print("index = ",index)
        
        ids, masks, tweet, offsets, sentiment_id, sentiment_target = self.get_input_data(row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets
        target['sentiment_id'] = sentiment_id
        target['sentiment_target'] = sentiment_target
        
        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx
            target['start_idx'] = start_idx
            target['end_idx'] = end_idx

        return data, target

    def __len__(self):
        return len(self.df)

    def sentiment_to_target(self, sentiment):
        targets = {'positive': 1, 'negative': 2, 'neutral': 0}
        return targets[sentiment]
    
    def get_input_data(self, row):
        try:
            tweet = " " + " ".join(row.text.lower().split())
        except:
            print(row)

        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(row.sentiment).ids

        ids = [0] + encoding.ids + [2]
        offsets = [(0, 0)] + encoding.offsets + [(0, 0)]
                
        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len
        
        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        sentiment_id = torch.tensor(sentiment_id)
        sentiment_target = torch.tensor(self.sentiment_to_target(row.sentiment))

        return ids, masks, tweet, offsets, sentiment_id, sentiment_target
        
    def get_target_idx(self, row, tweet, offsets):
        start_idx = 0
        end_idx = 0
        try:
            selected_text = " " +  " ".join(row.selected_text.lower().split())
            if len(selected_text) != selected_text.count(' '):
                len_st = len(selected_text) - 1
                idx0 = None
                idx1 = None
                for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
                    if " " + tweet[ind: ind+len_st] == selected_text:
                        idx0 = ind
                        idx1 = ind + len_st - 1
                        break

                char_targets = [0] * len(tweet)
                if idx0 != None and idx1 != None:
                    for ct in range(idx0, idx1 + 1):
                        char_targets[ct] = 1

                target_idx = []
                for j, (offset1, offset2) in enumerate(offsets):
                    if sum(char_targets[offset1: offset2]) > 0:
                        target_idx.append(j)

                start_idx = target_idx[0]
                end_idx = target_idx[-1]
        except:
            print("selected_text is empty with spaces")
            
        return start_idx, end_idx
