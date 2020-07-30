from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from easydict import EasyDict as edict
from collections import defaultdict
from torchvision import datasets, transforms
from singleton_decorator import SingletonDecorator


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
            raise KeyError(
                f'{class_name} is already registered in {self.name}')

        self._class_dict[class_name] = klass
        return klass


@SingletonDecorator
class Registry:
    def __init__(self):
        self.categories = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        for k, v in self.categories.items():
            format_str += f'(category[{k}]: {v})'

        return format_str

    def clear(self):
        self.categories.clear()

    def add(self, category, klass=None):
        if category not in self.categories:
            self.categories[category] = Category(category)

        if klass is not None:
            self.categories[category].add(klass)

    def register(self, category=''):
        def _register(klass):
            return self.add(category, klass)
        return _register

    def build_from_config(self, category, config, default_args=None):
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
        self.add(category)
        klass = self.categories[category].get(name)
        if klass is None:
            raise KeyError(f'{name} is not in the {category} registry')

        args = dict()
        if default_args is not None:
            args.update(default_args)
        if 'params' in config:
            args.update(config['params'])
        return klass(**args)

    def build_model(self, config, **kwargs):
        return self.build_from_config('model', config.model, kwargs)

    def build_loss_fn(self, config, **kwargs):
        return self.build_from_config('loss', config.loss, kwargs)

    def build_optimizer(self, config, **kwargs):
        return self.build_from_config('optimizer', config.optimizer, kwargs)

    def build_scheduler(self, config, **kwargs):
        return self.build_from_config('scheduler', config.scheduler, kwargs)

    def build_hooks(self, config):
        pass

    def build_transforms(self, config, **kwargs):
        if config.transforms.name == 'Compose':
            transfms = []
            for t in config.transforms.params:
                transfms.append(self.build_from_config('transform', t))

            return transforms.Compose(transfms)
        else:
            return self.build_from_config('transform', config.transforms),

    def build_dataloaders(self, config, **kwargs):
        dataloaders = []
        for split_config in config.dataset.splits:
            dataset_config = edict({'name': config.dataset.name,
                                    'params': config.dataset.params})
            dataset_config.params.update(split_config)

            transform = self.build_transforms(config)

            dataset = self.build_from_config('dataset', config.dataset, default_args={'transform': transform})

            is_train = dataset_config.params.train
            if is_train:
                batch_size = config.train.batch_size
            else:
                batch_size = config.evaluation.batch_size
            dataloader = DataLoader(dataset,
                                    shuffle=is_train,
                                    batch_size=batch_size,
                                    drop_last=is_train,
                                    num_workers=config.transforms.num_preprocessor,
                                    pin_memory=True)

            dataloaders.append({'mode': is_train,'dataloader': dataloader})
        return dataloaders


registry = Registry()

registry.add('loss', nn.NLLLoss)
registry.add('optimizer', optim.Adadelta)
registry.add('scheduler', optim.lr_scheduler.StepLR)
registry.add('dataset', datasets.MNIST)
registry.add('transform', transforms.ToTensor)
registry.add('transform', transforms.Normalize)