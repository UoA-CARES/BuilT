from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import logging
import numpy as np
from typing import Dict

from collections import defaultdict
from enum import Enum

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
        
        for k, v in log_dict.items():
            print(f'{k}: {v}')

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

    def __init__(self, writer: LogWriter, epoch: int, total_step: int):
        self.__writer = writer
        self.__mode: TrainMode = None
        self.__epoch: int = epoch
        self.__total_log_step: int = total_step
        self.__current_log_step: int = 0        
        self.__log_dict: Dict[str, any] = {}
        self.__aggregated_dict = defaultdict(list)
        self.__score: float = None

    def __is_numeric(self, value) -> bool:
        if isinstance(value, int) or isinstance(value, float) or isinstance(value, complex):
            return True

    def __initialized(self):
        if self.total_step is None:
            raise RuntimeError('total_step is not assigned.')
        if self.epoch is None:
            raise RuntimeError('epoch is not assigned.')
        if self.writer is None:
            raise RuntimeError('writer is not assigned.')

    def log(self, key: str, value: any, step: int):
        self.__initialized()

        self.__log_dict[key] = value
        if self.__is_numeric(value):
            if step == 0 or step <= len(self.__aggregated_dict[key]):
                self.__aggregated_dict[key].append(value)
            elif step  == len(self.__aggregated_dict[key]) - 1:
                self.__aggregated_dict[key][step] = value
            else:
                pass
        else:
            pass

    def log_dict(self, log_dict: Dict[str, any], step: int):
        self.__initialized()

        for key, value in log_dict.items():
            self.log(key, value, step)

    def aggregate(self):
        aggr = {f'avg_{key}':np.mean(value) for key, value in self.__aggregated_dict.items()}
        aggr['epoch'] = self.epoch
        self.__writer.log(aggr)

    @abc.abstractmethod
    def __log_extras(self, inputs: any, targets: any, outputs: any):
        self.score = 0

    def write(self, step: int, inputs=None, targets=None, outputs=None):
        assert self.__writer is not None        
        self.__log_extras(inputs, targets, outputs)
        self.__writer.log(self.__log_dict)
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
