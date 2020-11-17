from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import torch
import pandas as pd

from easydict import EasyDict as edict
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from built.trainer import Trainer
from built.builder import Builder


ex = Experiment('orsum')
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    description = 'Tweet Sentiment Classification'


@ex.main
def main(_run, _config):
    config = edict(_config)
    pprint.PrettyPrinter(indent=2).pprint(config)


@ex.command
def train(_run, _config):
    config = edict(_config)    
    pprint.PrettyPrinter(indent=2).pprint(config)

    builder = Builder()
    splitter = builder.build_splitter(config)
    df = pd.read_csv(splitter.csv_path)

    if not os.path.exists(config.train.dir):
        os.makedirs(config.train.dir)
        
    for i_fold in range(splitter.n_splits):
        print(f'Training start: {i_fold} fold')
        train_idx, val_idx = splitter.get_fold(i_fold)

        train_df = df.iloc[train_idx]
        train_csv_path = os.path.join(config.train.dir, str(i_fold) + '_train.csv')
        train_df.to_csv(train_csv_path)
        
        val_df = df.iloc[val_idx]
        val_csv_path = os.path.join(config.train.dir, str(i_fold) + '_val.csv')
        val_df.to_csv(val_csv_path)
        config.dataset.splits = []
        config.dataset.splits.append({'train': True, 'csv_path': train_csv_path})
        config.dataset.splits.append({'train': False, 'csv_path': val_csv_path})
        config.train.name = str(i_fold) + '_fold'
        
        tr = Trainer(config, builder)
        tr.run()
        print(f'Training end\n')


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic=True
    ex.run_commandline()
