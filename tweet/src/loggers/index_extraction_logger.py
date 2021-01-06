
import logging
import torch
import numpy as np

from sklearn import metrics
from built.logger import LoggerBase
from built.registry import Registry


@Registry.register(category="hooks")
class TweetIndexExtractionLogger(LoggerBase):
    def __call__(self, writer, split, outputs, labels, log_dict,
                 epoch, step=None, num_steps_in_epoch=None, data=None):

        if step is not None:
            assert num_steps_in_epoch is not None
            # log_step = epoch * 10000 + (step / num_steps_in_epoch) * 10000
            # log_step = int(log_step)
            log_step = (epoch + 1) * num_steps_in_epoch + step
        else:
            log_step = epoch

        for key, value in log_dict.items():
            if self.use_tensorboard:
                writer['tensorboard'].add_scalar(
                    f'{split}/{key}', value, log_step)

        
        log_dict.update({'epoch': epoch, 'log_step': log_step})
        keys = list(log_dict.keys())
        for k in keys:
            prefix = f'[{split}]'
            if prefix not in k:
                new_key = f'{prefix}_{k}'
                log_dict[new_key] = log_dict.pop(k)
                
        if self.use_wandb:
            writer['wandb'].log(log_dict)

        # if labels is not None and outputs is not None:
        #     start_idx = labels['start_idx']
        #     end_idx = labels['end_idx']
        #     start_pred = torch.softmax(
        #                 outputs[0], dim=1).cpu().detach().numpy()
        #     end_pred = torch.softmax(
        #                 outputs[1], dim=1).cpu().detach().numpy()

        #     writer['tensorboard'].add_pr_curve(
        #         f'pr_curve_{split}_start_idx', start_idx, start_pred, log_step)

        #     writer['tensorboard'].add_pr_curve(
        #         f'pr_curve_{split}_end_idx', end_idx, end_pred, log_step)
            
            
