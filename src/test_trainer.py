import unittest
import yaml

from easydict import EasyDict as edict
from sacred import Experiment
from trainer import Trainer

import models

yaml_config = """
dataset:
  name: "MNIST"
  params:
    root: "data"
    download: True

  splits:
    - train: True
    - train: False      

transform:
  name: "DefaultTransform"
  num_preprocessor: 8
  params:
    resize_to: (224, 224)

model:
  name: "Mnist"
  params:
    num_classes: 10

train:
  dir: "train_dirs/default"
  batch_size: 64
  num_epochs: 14

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

hooks:
  post_forward:
    name: "DefaultPostForwardHook"
"""


class TestTrainer(unittest.TestCase):

    def test_train(self):
        config = edict(yaml.load(yaml_config))
        tr = Trainer(config)
        tr.run()


if __name__ == '__main__':
    unittest.main()
