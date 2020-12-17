
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
class TweetSentimentClassificationModel(nn.Module):
    """BERT model for QA and classification tasks.

    Parameters
    ----------
    config : transformers.BertConfig. Configuration class for BERT.
    Returns
    -------
    classifier_logits : torch.Tensor with shape (batch_size, num_classes).
        Classification scores of each labels.
    """

    def __init__(self, transformer_type, transformer_path, drop_out_rate, num_classes):
        super().__init__()
        self.transformer_type = transformer_type
        self.transformer_path = transformer_path
        
        if self.transformer_type == 'roberta':
            model_config = transformers.RobertaConfig.from_pretrained(
                self.transformer_path, output_hidden_states=True)
            self.transformer = transformers.RobertaModel.from_pretrained(
                self.transformer_path, config=model_config)
        elif self.transformer_type == 'bert':
            model_config = transformers.BertConfig.from_pretrained(
                self.transformer_path, output_hidden_states=True)
            self.transformer = transformers.BertModel.from_pretrained(
                self.transformer_path, config=model_config)
        
        self.drop_out = nn.Dropout(drop_out_rate)
        #self.classifier = nn.Linear(768, num_classes)
        self.classifier = nn.Linear(model_config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, position_ids=None, head_mask=None):
        last_hidden_states, pooled_output, hidden_states = self.transformer(
            input_ids, attention_mask=attention_mask)

        # classification
        pooled_output = self.drop_out(pooled_output)
        classifier_logits = self.classifier(pooled_output)

        return last_hidden_states, classifier_logits
