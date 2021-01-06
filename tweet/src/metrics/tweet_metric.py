from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import logging

import numpy as np
import torch

from sklearn import metrics
from built.metric import MetricBase
from built.registry import Registry


@Registry.register(category='hooks')
class TweetMetric(MetricBase):
    def __call__(self, outputs, targets, data=None, is_train=False, device='cpu'):
        outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
        outputs = np.argmax(outputs, axis=1)

        targets = targets['sentiment_target']
        targets = targets.cpu().detach().numpy()

        accuracy = metrics.accuracy_score(targets, outputs)
        precision = metrics.precision_score(
            targets, outputs, average='micro')

        recall = metrics.recall_score(
            targets, outputs, average='micro')

        f1_score = metrics.f1_score(
            targets, outputs, average='micro')

        return {
            'score': accuracy, 
            'accuracy': accuracy, 
            'precision': precision, 
            'recall': recall, 
            'f1_score': f1_score}
