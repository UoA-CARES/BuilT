from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc


class LossBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, outputs, targets, data=None, is_train=False, device='cpu'):
        pass


class DefaultLoss(LossBase):
    def __call__(self, loss_fn, outputs, targets, data, is_train, device='cpu'):
        if isinstance(outputs, dict):
            loss = loss_fn(input=outputs['logits'], target=targets)
        else:
            loss = loss_fn(input=outputs, target=targets)

        return loss


        
