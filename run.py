from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pprint
import torch

from easydict import EasyDict as edict
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds


ex = Experiment('orsum')
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def cfg():
    description = ''


@ex.main
def main(_run, _config):
    config = edict(_config)
    pprint.PrettyPrinter(indent=2).pprint(config)


@ex.command
def train(_run, _config):
    config = edict(_config)    
    pprint.PrettyPrinter(indent=2).pprint(config)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic=True
    ex.run_commandline()
