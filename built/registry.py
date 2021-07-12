from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from collections import defaultdict
from torchvision import datasets, transforms

from built.singleton_decorator import SingletonDecorator
from built.forward_hook import DefaultPostForwardHook
from built.metric import DefaultMetric
from built.logger import DefaultLogger
from built.models.mnist import Mnist

class Category:
    def __init__(self, name):
        self._name = name
        self._class_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(name={self._name}, items={list(self._class_dict.keys())})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def class_dict(self):
        return self._class_dict

    def get(self, key: str):
        return self._class_dict.get(key, None)

    def add(self, klass):
        """add a callable class.

        Args:
            klass: callable class to be registered
        """
        if not callable(klass):
            raise ValueError(f'object must be callable')

        class_name = klass.__name__
        if class_name in self._class_dict:
            print(f'{class_name} is already registered in {self.name}')
        else:
            self._class_dict[class_name] = klass
        return klass


class Registry:

    categories = dict()

    # def __repr__(self):
    #     format_str = self.__class__.__name__
    #     for k, v in self.categories.items():
    #         format_str += f'(category[{k}]: {v})'

    #     return format_str

    @classmethod
    def clear(cls):
        cls.categories.clear()

    @classmethod
    def add(cls, category, klass=None):
        if category not in cls.categories:
            cls.categories[category] = Category(category)

        if klass is not None:
            cls.categories[category].add(klass)

    @classmethod
    def register(cls, category=''):
        def _register(klass):
            return cls.add(category, klass)
        return _register

    @classmethod
    def build_from_config(cls, category, config, default_args=None):
        """Build a callable object from configuation dict.

        Args:
            category: The name of category to search the name from.
            config (dict): Configuration dict. It should contain the key "name".            
            default_args (dict, optional): Default initialization argments.
        """
        assert isinstance(config, dict) and 'name' in config
        assert isinstance(default_args, dict) or default_args is None

        name = config['name']
        name = name.replace('-', '_')
        cls.add(category)
        klass = cls.categories[category].get(name)
        if klass is None:
            raise KeyError(f'{name} is not in the {category} registry')

        args = dict()
        if default_args is not None:
            args.update(default_args)
        if 'params' in config:
            args.update(config['params'])
        return klass(**args)