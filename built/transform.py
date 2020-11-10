from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from torchvision import transforms

class TransformBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, transforms):
        pass
