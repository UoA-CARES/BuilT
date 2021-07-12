from __future__ import print_function

import numpy as np
import torch


class EarlyStopper:
    def __init__(self, patience=7, mode="max", delta=0):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.comp = None
        self.delta = delta
        if self.mode == "min":
            self.best_score = np.Inf
            self.comp = self.min_compare
        else:
            self.best_score = -np.Inf
            self.comp = self.max_compare
            
    def min_compare(self, l, r):
        return l < r
    
    def max_compare(self, l, r):
        return l > r
            
    def __call__(self, epoch_score):
        assert(epoch_score not in [-np.inf, np.inf, -np.nan, np.nan])

        save_checkpoint = False
        if self.comp(epoch_score, self.best_score + self.delta):
            print(
                f'Validation score improved ({self.best_score} --> {epoch_score})')
            self.best_score = epoch_score
            save_checkpoint = True
            self.counter = 0
        else:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(
                self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop, save_checkpoint
