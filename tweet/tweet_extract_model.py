
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


ROBERTA_PATH = '../input/roberta-base/'

@Registry.register(category="model")
class TweetExtractModel(transformers.BertPreTrainedModel):
    """BERT model for QA and classification tasks.

    Parameters
    ----------
    config : transformers.BertConfig. Configuration class for BERT.
    Returns
    -------
    classifier_logits : torch.Tensor with shape (batch_size, num_classes).
        Classification scores of each labels.
    """

    def __init__(self, conf):
        super(TweetExtractModel, self).__init__(conf)
        self.roberta = transformers.RobertaModel.from_pretrained(
            ROBERTA_PATH, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        hidden_states, pooled_output, _ = self.roberta(input_ids,
                                                       attention_mask=attention_mask,
                                                       token_type_ids=token_type_ids)

        # classification
        pooled_output = self.drop_out(pooled_output)
        classifier_logits = self.classifier(pooled_output)

        return hidden_states, classifier_logits
