from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import logging
import torch


class ForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def to_cuda(self, inputs, model, device):
        model = model.to(device)
        if isinstance(inputs, dict):
            for k in inputs.keys():
                if torch.is_tensor(inputs[k]):
                    inputs[k] = inputs[k].to(device)
        else:
            if torch.is_tensor(inputs):
                inputs = inputs.to(device)
        
        return inputs, model

    @abc.abstractmethod
    def forward(self, inputs, model, is_train):
        pass

    @abc.abstractmethod
    def __call__(self, model, inputs, targets=None, data=None, is_train=False, device='cpu'):
        if device != 'cpu':
            inputs, model = self.to_cuda(inputs, model, device)

        model.zero_grad()
        outputs = self.forward(inputs, model, is_train)
        return outputs


class DefaultForwardHook(ForwardHookBase):
    def __call__(self, model, inputs, targets=None, data=None, is_train=False, device='cpu'):
        logging.debug("Default forward hook is called")
        return model(inputs)


class PostForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, outputs, inputs=None, targets=None, data=None, is_train=False):
        pass


class DefaultPostForwardHook(PostForwardHookBase):
    def __call__(self, outputs, inputs=None, targets=None, data=None, is_train=False):
        logging.info("Default post forward hook is called")
        return outputs
