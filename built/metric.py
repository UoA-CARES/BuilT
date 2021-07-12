from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import logging

import numpy as np
import torch

from typing import Dict

class MetricBase(object):
    __metaclass__ = abc.ABCMeta

    class MetricStore(object):
        def __init__(self):
            self.__store: Dict[str, float] = {}
        
        def add(self, key: str, value: float):
            assert key is not None
            assert value is not None

            if key in self.__store.keys():
                raise KeyError(f'{key} already exists.')

            self.__store[key] = value

        def update(self, key: str, value: float):
            assert key is not None
            assert value is not None

            if key in self.__store.keys():
                self.__store[key] = value
            else:
                raise KeyError(f'{key} does not exist.')

        def get(self) -> Dict[str, float]:
            return self.__store

    def __init__(self):
        self.store = self.MetricStore()
    
    @abc.abstractmethod
    def calc(self, outputs, targets, extra_data=None, is_train=False, device='cpu'):
        print('test')
        pass
    
    def add(self, key: str, value: float):
        try:
            self.store.add(key, value)
        except KeyError:
            self.store.update(key, value)

    def calculate(self, outputs, targets, extra_data=None, is_train=False, device='cpu') -> Dict[str, float]:
        self.calc(outputs, targets, extra_data, is_train, device)
        return self.store.get()


class DefaultMetric(MetricBase):
    def calc(self, outputs, targets, daextra_datata=None, is_train=False, device='cpu'):
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

        self.add('accuracy', accuracy)
        self.add('score', accuracy)
