import sys
import os
import yaml

from easydict import EasyDict as edict

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from src.trainer import Trainer

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
    use_wandb: True
"""

hyperparameter_defaults = dict(
    batch_size=1,
    learning_rate=0.001,
    epochs=2,
)

config = edict(yaml.load(yaml_config))
config.train.batch_size = hyperparameter_defaults['batch_size']
config.optimizer.params.lr = hyperparameter_defaults['learning_rate']
config.train.num_epochs = hyperparameter_defaults['epochs']

trainer = Trainer(config, hyperparameter_defaults)
trainer.run()
