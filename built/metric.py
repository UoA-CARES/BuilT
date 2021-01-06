from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import logging

import numpy as np
import torch


class MetricBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, outputs, targets, data=None, is_train=False, device='cpu'):
        pass


class DefaultMetric(MetricBase):
    def __call__(self, outputs, targets, data=None, is_train=False, device='cpu'):
        logging.debug("Default metric is called")
        if isinstance(outputs, dict):
            logits = outputs['logits']
        else:
            logits = outputs

        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().detach().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().detach().numpy()            

        assert len(logits.shape) == 2
        predicts = np.argmax(logits, axis=1)
        correct = np.sum((predicts == labels).astype(int))
        total = predicts.shape[0]
        accuracy = 100. * correct / total

        return {'score': accuracy, 'accuracy': accuracy}
