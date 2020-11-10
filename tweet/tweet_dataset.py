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


sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
sentiment_tar = {'positive': 1, 'negative': 2, 'neutral': 0}

def process_data(tweet, selected_text, sentiment, tokenizer, max_len, do_ques_ans=False):

    # FIND TEXT / SELECTED_TEXT OVERLAP
    tweet = " " + " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())

    start_idx = tweet.find(selected_text)
    end_idx = start_idx + len(selected_text)

    char_targets = [0] * len(tweet)
    if start_idx != None and end_idx != None:
        for ct in range(start_idx, end_idx):
            char_targets[ct] = 1

    tok = tokenizer.encode(tweet)
    tweet_ids = tok.ids

    # OFFSETS, CHAR CENTERS
    tweet_offsets = tok.offsets
    char_centers = [(offset[0] + offset[1]) / 2 for offset in tweet_offsets]

    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)

#     print(target_idx)

    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    stok = [sentiment_id[sentiment]]

    input_ids = [0] + tweet_ids + [2]
    token_type_ids = [0] + [0] * (len(tweet_ids) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] + tweet_offsets + [(0, 0)]
    targets_start += 1
    targets_end += 1

    if do_ques_ans:
        input_ids = [0] + stok + [2] + [2] + tweet_ids + [2]
        token_type_ids = [0, 0, 0, 0] + [0] * (len(tweet_ids) + 1)
        mask = [1] * len(token_type_ids)
        tweet_offsets = [(0, 0)] * 4 + tweet_offsets + [(0, 0)]
        targets_start += 4
        targets_end += 4

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([1] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
#         print(char_centers)
        char_centers = char_centers + ([0] * padding_length)

    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets,
        'char_cent': char_centers,
        'sentiment_tar': sentiment_tar[sentiment]
    }


@Registry.register(category='dataset')
class TweetDataset:
    def __init__(self, max_len, roberta_path, tweet=None, sentiment=None, selected_text=None, train=False):
        print('TweetDataset!!!')
        self.max_len = max_len
        self.roberta_path = roberta_path
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = tokenizers.ByteLevelBPETokenizer(
                        vocab_file=self.roberta_path+'vocab.json',
                        merges_file=self.roberta_path+'merges.txt',
                        lowercase=True,
                        add_prefix_space=True
                    )

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item],
            self.selected_text[item],
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets': torch.tensor(data["offsets"], dtype=torch.long),
            'char_cent': torch.tensor(data['char_cent'], dtype=torch.long),
            'sentiment_tar': data['sentiment_tar']
        }