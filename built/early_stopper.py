from __future__ import print_function

import numpy as np
import torch


class EarlyStopper:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score):
        assert(epoch_score not in [-np.inf, np.inf, -np.nan, np.nan])
        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        save_checkpoint = False
        if self.best_score is None or score >= (self.best_score + self.delta):
            print(
                f'Validation score improved ({self.val_score} --> {epoch_score})')
            self.best_score = score
            self.val_score = epoch_score
            save_checkpoint = True
            self.counter = 0
        else:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(
                self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop, save_checkpoint
