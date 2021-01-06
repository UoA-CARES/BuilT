
import os
import logging
import torch
import numpy as np

from sklearn import metrics
from built.logger import LoggerBase
from built.registry import Registry


@Registry.register(category="hooks")
class TweetLogger(LoggerBase):
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

        if isinstance(outputs, list):
            outputs = torch.cat(tuple(outputs))
            # sentiment_targets = []
            # for d in labels:
            #     sentiment_targets.append(d['sentiment_target'])
            sentiment_targets = labels['sentiment_target']
            sentiment_targets = torch.cat(tuple(sentiment_targets))

            preds = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
            preds = np.argmax(preds, axis=1)

            y = sentiment_targets.numpy()

            fpr, tpr, thresholds = metrics.roc_curve(
                y, preds, pos_label=2)

            auc = metrics.auc(fpr, tpr)

            log_dict.update({'auc': auc})

        log_dict.update({'epoch': epoch, 'log_step': log_step})
        keys = list(log_dict.keys())
        for k in keys:
            prefix = f'[{split}]'
            if prefix not in k:
                new_key = f'{prefix}_{k}' 
                log_dict[new_key] = log_dict.pop(k)
            
        if self.use_wandb:    
            
            writer['wandb'].log(log_dict)

        if self.use_tensorboard:
            if labels is not None and outputs is not None:
                labels = labels['sentiment_target']
                outputs = torch.sigmoid(
                    outputs).cpu().detach().numpy().tolist()
                outputs = np.argmax(outputs, axis=1)

                writer['tensorboard'].add_pr_curve(
                    f'pr_curve_{split}', labels, outputs, log_step)

                y = labels.numpy()
                pred = outputs
                fpr, tpr, thresholds = metrics.roc_curve(
                    labels.numpy(), outputs, pos_label=2)
                auc = metrics.auc(fpr, tpr)

                writer['tensorboard'].add_scalar(
                    f'auc_curve_{split}', auc, log_step)
