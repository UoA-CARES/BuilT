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

from torch.nn import functional as F
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


class TweetDatasetBase(torch.utils.data.Dataset):
    def __init__(self, model_path, csv_path, transformer_type='roberta', train=False, split='train', max_len=96):
        df = pd.read_csv(csv_path)
        
        
        self.df = df.dropna().reset_index(drop=True)
        self.max_len = max_len
        self.labeled = 'selected_text' in self.df
        self.transformer_type = transformer_type
        self.tokenizer = self.get_tokenizer(model_path)

    def get_tokenizer(self, model_path):
        tokenizer = None
        if 'roberta' in self.transformer_type:
            tokenizer = tokenizers.ByteLevelBPETokenizer(
                vocab_file=model_path+'vocab.json',
                merges_file=model_path+'merges.txt',
                lowercase=True,
                add_prefix_space=True)
        elif 'bert' in self.transformer_type:
            vocab_path = os.path.join(model_path, 'vocab.txt')
            tokenizer = tokenizers.BertWordPieceTokenizer(
                vocab_path,
                lowercase=True
            )
        else:
            raise RuntimeError(f'{self.transformer_type} is not supported')

        return tokenizer

    def __getitem__(self, index):
        data = {}
        target = {}

        row = self.df.iloc[index]

        ids, masks, tweet, offsets, sentiment_id, sentiment_target, char_centers = self.get_input_data(
            row)
        data['ids'] = ids
        data['masks'] = masks
        data['tweet'] = tweet
        data['offsets'] = offsets
        data['sentiment_id'] = sentiment_id
        data['sentiment_target'] = sentiment_target
        data['char_centers'] = char_centers
        target['sentiment_id'] = sentiment_id
        target['sentiment_target'] = sentiment_target

        if self.labeled:
            start_idx, end_idx, selected_text = self.get_target_idx(
                row, tweet, offsets)
            data['start_idx'] = start_idx
            data['end_idx'] = end_idx
            data['selected_text'] = selected_text
            target['start_idx'] = start_idx
            target['end_idx'] = end_idx
            target['selected_text'] = selected_text
            target['offsets'] = offsets
            target['tweet'] = tweet

        return data, target

    def __len__(self):
        return len(self.df)

    def sentiment_to_target(self, sentiment):
        targets = {'positive': 1, 'negative': 2, 'neutral': 0}
        return targets[sentiment]

    def encode_ids(self, encoding, row=None):
        pass

    def encode_offsets(self, encoding):
        pass

    def get_input_data(self, row):
        try:
            tweet = " " + " ".join(row.text.lower().split())
        except:
            raise RuntimeError(f'{row}')

        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(row.sentiment).ids

        ids = self.encode_ids(encoding, row)
        offsets = self.encode_offsets(encoding)

        char_centers = [(x[0] + x[1]) / 2 for x in encoding.offsets]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len
            char_centers += [0] * pad_len

        ids = torch.tensor(ids)
        masks = torch.where(ids != 1, torch.tensor(1), torch.tensor(0))
        offsets = torch.tensor(offsets)
        sentiment_id = torch.tensor(sentiment_id)
        sentiment_target = torch.tensor(
            self.sentiment_to_target(row.sentiment))
        char_centers = torch.tensor(char_centers)
        return ids, masks, tweet, offsets, sentiment_id, sentiment_target, char_centers

    def get_target_idx(self, row, tweet, offsets):
        start_idx = 0
        end_idx = 0
        try:
            # selected_text = " " + " ".join(row.selected_text.lower().split())
            tweet = " " + " ".join(str(tweet).split())
            selected_text = " ".join(row.selected_text.lower().split())

            if len(selected_text) != selected_text.count(' '):
                start_idx = tweet.find(selected_text)
                end_idx = start_idx + len(selected_text)

                char_targets = [0] * len(tweet)
                if start_idx != None and end_idx != None:
                    for ct in range(start_idx, end_idx):
                        char_targets[ct] = 1

                target_idx = []
                for j, (offset1, offset2) in enumerate(offsets):
                    if sum(char_targets[offset1: offset2]) > 0:
                        target_idx.append(j)

                start_idx = target_idx[0]
                end_idx = target_idx[-1]
        except:
            print("selected_text is empty with spaces")

        return start_idx, end_idx, selected_text
    # def get_target_idx(self, row, tweet, offsets):
    #     start_idx = 0
    #     end_idx = 0
    #     try:
    #         selected_text = " " + " ".join(row.selected_text.lower().split())
    #         if len(selected_text) != selected_text.count(' '):
    #             len_st = len(selected_text) - 1
    #             idx0 = None
    #             idx1 = None
    #             for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
    #                 if " " + tweet[ind: ind+len_st] == selected_text:
    #                     idx0 = ind
    #                     idx1 = ind + len_st - 1
    #                     break

    #             char_targets = [0] * len(tweet)
    #             if idx0 != None and idx1 != None:
    #                 for ct in range(idx0, idx1 + 1):
    #                     char_targets[ct] = 1

    #             target_idx = []
    #             for j, (offset1, offset2) in enumerate(offsets):
    #                 if sum(char_targets[offset1: offset2]) > 0:
    #                     target_idx.append(j)

    #             start_idx = target_idx[0]
    #             end_idx = target_idx[-1]
    #     except:
    #         print("selected_text is empty with spaces")

    #     return start_idx, end_idx, selected_text
