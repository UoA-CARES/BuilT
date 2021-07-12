from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import logging
import numbers
import torch
import numpy as np
from typing import Dict

from collections import defaultdict
from enum import Enum
from numpy.lib.arraysetops import isin

class TrainMode(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

# for key, value in self.__log_dict.items():
#     if self.use_tensorboard:
#         writer['tensorboard'].add_scalar(f'{self.mode}/{key}', value, self.step)
        
# if self.use_wandb:
#     self.__log_dict.update({'epoch': epoch, 'mode':self.mode})
#     writer['wandb'].log(self.__log_dict)

class LogWriter(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def log(self, log_dict: Dict[str, any]):
        assert log_dict is not None
        
        # for k, v in log_dict.items():
            # print(f'{k}: {v}')

import wandb

class WandbWriter(LogWriter):
    def __init__(self, run=None, project: str=None, group: str=None, reinit=True):
        super().__init__()
        self.run = run
        if self.run is None:
            assert project is not None
            assert group is not None

            self.run = wandb.init(project=project, group=group, reinit=reinit)

    def log(self, log_dict: Dict[str, any]):
        self.run.log(log_dict)


class LoggerBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, writer: LogWriter, epoch: int, total_step: int, is_train: bool):
        self.__writer = writer
        self.__mode: TrainMode = None
        self.__epoch: int = epoch
        self.__total_log_step: int = total_step
        self.__current_log_step: int = 0        
        self.__log_dict: Dict[str, any] = {}
        self.__aggregated_dict = defaultdict(list)
        self.__score: float = None
        self.__loss: float = None
        self.__batch_size: float = None
        self.is_train = is_train
        self.total_score = 0
        self.total_batch_size = 0
        self.avg = 0

    def __is_numeric(self, value) -> bool:
        # if isinstance(value, int) or isinstance(value, float) or isinstance(value, complex):
        if isinstance(value, numbers.Number):
            return True
        else:
            return False

    def __initialized(self):
        if self.total_step is None:
            raise RuntimeError('total_step is not assigned.')
        if self.epoch is None:
            raise RuntimeError('epoch is not assigned.')
        if self.writer is None:
            raise RuntimeError('writer is not assigned.')

    def log(self, key: str, value: any, step: int):
        self.__initialized()
        
        if torch.is_tensor(value):
            value = value.cpu().detach().item()

        if key == 'score':
            self.total_score += value * self.batch_size
            self.total_batch_size += self.batch_size
            self.avg = self.total_score / self.total_batch_size
            # print(self.avg, value, self.total_score, self.total_batch_size)

        self.__log_dict[key] = value
        if self.__is_numeric(value):
            if step == 0 or step <= len(self.__aggregated_dict[key]):
                self.__aggregated_dict[key].append(value)
            elif step == len(self.__aggregated_dict[key]) - 1:
                self.__aggregated_dict[key][step] = value
            else:
                pass

            self.__log_dict['avg_' + key] = np.mean(self.__aggregated_dict[key])
        else:
            pass

    def log_dict(self, log_dict: Dict[str, any], step: int):
        self.__initialized()

        for key, value in log_dict.items():
            self.log(key, value, step)        

    def aggregate(self):
        aggr = {f'avg_{key}':np.mean(value) for key, value in self.__aggregated_dict.items()}
        aggr['epoch'] = self.epoch
        
        if 'avg_score' in aggr:
            self.score = aggr['avg_score']
        else:
            self.score = None

        suffix = '_train' if self.is_train else '_val'
        aggr = { (k+suffix) : v for k, v in aggr.items() }

        self.__writer.log(aggr)

    @abc.abstractmethod
    def log_extras(self, inputs: any, targets: any, outputs: any):
        pass

    def write(self, step: int, inputs=None, targets=None, outputs=None):
        assert self.__writer is not None        
        self.log_extras(inputs, targets, outputs)

        # if 'avg_score' in self.__log_dict:
        avg = self.__log_dict['avg_score']
        self.score = self.avg
        # print(avg - self.score)
        # else:
            # self.score = None

        if 'avg_loss' in self.__log_dict:
            self.loss = self.__log_dict['avg_loss']
        else:
            self.loss = None

        suffix = '_train' if self.is_train else '_val'
        log_dict = { (k+suffix) : v for k, v in self.__log_dict.items() }

        self.__writer.log(log_dict)
        self.__step_up()
    
    def __step_up(self):
        self.step += 1
    
    @property
    def step(self) -> int:
        return self.__current_log_step

    @step.setter
    def step(self, step: int):
        self.__current_log_step = step

    @property
    def epoch(self) -> int:
        return self.__epoch

    @epoch.setter
    def epoch(self, epoch: int):
        self.__epoch = epoch

    @property
    def total_step(self) -> int:
        return self.__total_log_step

    @total_step.setter
    def total_step(self, step: int):
        self.__total_log_step = step

    @property
    def mode(self) -> TrainMode:
        return self.__mode
    
    @mode.setter
    def mode(self, mode: TrainMode):
        self.__mode = mode

    @property
    def writer(self) -> LogWriter:
        return self.__writer
    
    @writer.setter
    def writer(self, writer: LogWriter):
        self.__writer = writer

    @property
    def score(self) -> float:
        return self.__score

    @score.setter
    def score(self, score: float):
        self.__score = score

    @property
    def loss(self) -> float:
        return self.__loss

    @loss.setter
    def loss(self, loss: float):
        self.__loss = loss

    @property
    def batch_size(self) -> float:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: float):
        self.__batch_size = batch_size



class DefaultLogger(LoggerBase):
    def __call__(self, writer, split, outputs, labels,
                 epoch, step=None, num_steps_in_epoch=None, data=None):
        logging.debug("Default logger is called")

        if step is not None:
            assert num_steps_in_epoch is not None
            log_step = epoch * 10000 + (step / num_steps_in_epoch) * 10000
            log_step = int(log_step)
        else:
            log_step = epoch
            
        if labels is not None and outputs is not None:
            writer['tensorboard'].add_pr_curve(
                'pr_curve', labels, outputs, log_step)
