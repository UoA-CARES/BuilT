from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import logging


class LoggerBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, use_tensorboard=None, use_wandb=None):
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

    @abc.abstractmethod
    def __call__(self, writer, split, outputs, labels, log_dict,
                 epoch, step=None, num_steps_in_epoch=None, data=None):
        pass

class DefaultLogger(LoggerBase):
    def __call__(self, writer, split, outputs, labels, log_dict,
                 epoch, step=None, num_steps_in_epoch=None, data=None):
        logging.debug("Default logger is called")

        if step is not None:
            assert num_steps_in_epoch is not None
            log_step = epoch * 10000 + (step / num_steps_in_epoch) * 10000
            log_step = int(log_step)
        else:
            log_step = epoch

        for key, value in log_dict.items():
            if self.use_tensorboard:
                writer['tensorboard'].add_scalar(f'{split}/{key}', value, log_step)
                
        if self.use_wandb:
            log_dict.update({'epoch': epoch, 'mode':split})
            writer['wandb'].log(log_dict)
            
        if labels is not None and outputs is not None:
            writer['tensorboard'].add_pr_curve(
                'pr_curve', labels, outputs, log_step)
