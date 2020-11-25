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
from built.registry import Registry


@Registry.register(category="model")
class TweetIndexExtractModel(nn.Module):
    def __init__(self, transformer_type, transformer_path, drop_out_rate):
        super().__init__()

        self.transformer_type = transformer_type
        self.transformer_path = transformer_path
        model_config = transformers.RobertaConfig.from_pretrained(
            self.transformer_path, output_hidden_states=True)
        self.roberta = transformers.RobertaModel.from_pretrained(
            self.transformer_path, config=model_config)

        self.dropout = nn.Dropout(drop_out_rate)
        self.fc = nn.Linear(model_config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

        self.conv0 = nn.Conv1d(model_config.hidden_size, 256, 3, padding=1)
        self.conv1 = nn.Conv1d(256, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, 3, padding=1)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, input_ids, attention_mask):
        _, _, hs = self.roberta(input_ids, attention_mask)

        x = torch.stack([hs[-1], hs[-2], hs[-3]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = self.conv0(x.transpose(1, 2))
        x = nn.functional.leaky_relu(x, 0.2, True)
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x, 0.2, True)
        x = self.conv2(x)
        x = self.fc2(x.transpose(1, 2))
        x = self.fc3(x)

        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits