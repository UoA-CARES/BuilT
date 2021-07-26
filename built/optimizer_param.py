from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc


class OptimizerParamBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, named_parameters):
        pass