
import sys
import pkgutil
import torch.nn as nn
import torch.optim as optim
import yaml

from easydict import EasyDict as edict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .registry import Registry

from .forward_hook import DefaultPostForwardHook
from .metric import DefaultMetric
from .logger import DefaultLogger
from .models.mnist import Mnist

class Builder(object):
    def __init__(self):
        # self.r = Registry()
        self.load_all_modules_from_dir('.')
        self.regist_defaults()

    def load_all_modules_from_dir(self, dirname):
        for importer, package_name, _ in pkgutil.iter_modules([dirname]):
            print(f'{package_name}')
            if package_name in sys.modules:
                print(package_name)
            if package_name not in sys.modules and package_name != 'main':
                module = importer.find_module(package_name).load_module(package_name)
                print(f'{module} is loaded!!!!!!!!!!!!!!!!11')

    def regist_defaults(self):
        Registry.add('loss', nn.NLLLoss)
        Registry.add('optimizer', optim.Adadelta)
        Registry.add('scheduler', optim.lr_scheduler.StepLR)
        Registry.add('dataset', datasets.MNIST)
        Registry.add('transform', transforms.ToTensor)
        Registry.add('transform', transforms.Normalize)
        Registry.add('hooks', DefaultPostForwardHook)
        Registry.add('hooks', DefaultMetric)
        Registry.add('hooks', DefaultLogger)
        Registry.add('model', Mnist)

    def build_model(self, config, **kwargs):
        return Registry.build_from_config('model', config.model, kwargs)

    def build_loss_fn(self, config, **kwargs):
        return Registry.build_from_config('loss', config.loss, kwargs)

    def build_optimizer(self, config, **kwargs):
        return Registry.build_from_config('optimizer', config.optimizer, kwargs)

    def build_scheduler(self, config, **kwargs):
        return Registry.build_from_config('scheduler', config.scheduler, kwargs)

    def build_forward_hook(self, config, **kwargs):
        return Registry.build_from_config('hooks', config.forward_hook, kwargs)

    def build_post_forward_hook(self, config, **kwargs):
        return Registry.build_from_config('hooks', config.post_forward_hook, kwargs)

    def build_metric_fn(self, config, **kwargs):
        return Registry.build_from_config('hooks', config.metric_hook, kwargs)

    def build_logger_fn(self, config, **kwargs):
        return Registry.build_from_config('hooks', config.logger_hook, kwargs)

    def build_transforms(self, config, **kwargs):
        if config.transforms.name == 'Compose':
            transfms = []
            for t in config.transforms.params:
                transfms.append(Registry.build_from_config('transform', t))

            return transforms.Compose(transfms)
        elif config.transforms.name == '':
            return None
        else:
            return Registry.build_from_config('transform', config.transforms),

    def build_dataloaders(self, config, **kwargs):
        dataloaders = []
        for split_config in config.dataset.splits:
            dataset_config = edict({'name': config.dataset.name,
                                    'params': config.dataset.params})
            dataset_config['params'].update(split_config)

            transform = self.build_transforms(config)

            dataset = Registry.build_from_config('dataset', config.dataset, default_args={})

            is_train = dataset_config['params'].train
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
    
    def build_sweep(self, config):
        with open(r'config.wandb.sweep.yaml') as file:
            hyperparam = yaml.load(file, Loader=yaml.FullLoader)
            
    
