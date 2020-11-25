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
class TweetIndexExtractionMetric(MetricBase):
    def __call__(self, outputs, targets, is_train, split):
        start_idx = targets['start_idx'].cpu().detach().numpy()
        end_idx = targets['end_idx'].cpu().detach().numpy()
        
        start_pred = torch.softmax(
                        outputs[0], dim=1).cpu().detach().numpy()
        start_pred = np.argmax(start_pred, axis=1)
        
        end_pred = torch.softmax(
                        outputs[1], dim=1).cpu().detach().numpy()
        end_pred = np.argmax(end_pred, axis=1)
        
        start_idx_accuracy = metrics.accuracy_score(start_idx, start_pred)
        end_idx_accuracy = metrics.accuracy_score(end_idx, end_pred)
        
        score = (start_idx_accuracy + end_idx_accuracy) / 2.0
        
        return {'score': score, 'start_idx_accuracy': start_idx_accuracy, 'end_idx_accuracy': end_idx_accuracy}
