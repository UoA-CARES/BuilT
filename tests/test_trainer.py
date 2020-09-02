import unittest
import yaml

from easydict import EasyDict as edict
from sacred import Experiment

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.singleton_decorator import SingletonDecorator
from src.forward_hook import DefaultPostForwardHook
from src.metric import DefaultMetric
from src.logger import DefaultLogger
from src.registry import Registry

from src.trainer import Trainer
from src.models.mnist import Mnist

yaml_config = """
wandb:
  sweep:
    name: "Sweep"
    use: True
    yaml: "sweep.yaml"
    
dataset:
  name: "MNIST"
  params:
    root: "data"
    download: True

  splits:
    - train: True
    - train: False      

transforms:
  name: "Compose"
  num_preprocessor: 8
  params:
    - ToTensor:
      name: "ToTensor"
    - Normalize:
      name: "Normalize"
      params:
        mean: !!python/tuple [0.1307, ]
        std: !!python/tuple [0.3081, ]


model:
  name: "Mnist"
  params:
    num_classes: 10

train:
  dir: "train_dirs/default"
  batch_size: 64
  num_epochs: 1
  gradient_accumulation_step: 1

evaluation:
  batch_size: 64

loss:
  name: "NLLLoss"

optimizer:
  name: "Adadelta"
  params:
    lr: 1.0

scheduler:
  name: "StepLR"
  params:
    step_size: 1
    gamma: 0.7

post_forward_hook:
  name: "DefaultPostForwardHook"

metric_hook:
  name: "DefaultMetric"

logger_hook:
  name: "DefaultLogger"
  params:
    use_tensorboard: True
    use_wandb: False
"""


class TestTrainer(unittest.TestCase):

    def test_train(self):
        print('--------------------------------------------------------------------')
        config = edict(yaml.load(yaml_config))
        tr = Trainer(config)
        tr.run()


if __name__ == '__main__':
    unittest.main()
