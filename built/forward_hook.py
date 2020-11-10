from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import logging


class ForwardHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, model, inputs, targets=None, data=None, is_train=False):
        pass


class DefaultForwardHook(ForwardHookBase):
    def __call__(self, model, inputs, targets=None, data=None, is_train=False):
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
