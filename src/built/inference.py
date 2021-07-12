import os
import math
import logging
import torch
import tqdm
import numpy as np
import pandas as pd
import wandb

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from collections import defaultdict

from built.builder import Builder
from built.checkpoint_manager import CheckpointManager
from built.early_stopper import EarlyStopper
from built.logger import LogWriter, WandbWriter


class Inference(object):
    def __init__(self, config, builder, working_dir=None):
        self.config = config
        self.builder = builder        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.working_dir = working_dir       
        self.cm = CheckpointManager(self.working_dir)
        self.build_classes()

    def build_classes(self):
        # build dataloaders
        self.dataloaders = self.builder.build_dataloaders(self.config)

        # build model
        self.model = self.builder.build_model(self.config)
        self.model = self.model.to(self.device)

        # build hooks
        self.forward_hook = self.builder.build_forward_hook(self.config)
        self.post_forward_hook = self.builder.build_post_forward_hook(self.config)

        # build optimizer
        # if 'no_bias_decay' in self.config.train and self.config.train.no_bias_decay:
        #     group_decay, group_no_decay = group_weight(self.model)
        #     params = [{'params': group_decay}, {
        #         'params': group_no_decay, 'weight_decay': 0.0}]
        # else:
        #     params = self.model.parameters()

        # total_steps = None
        # for d in self.dataloaders:
        #     is_train = d['mode']
        #     if is_train:
        #         total_steps = self.calc_steps(d['dataloader'], True)

        # self.optimizer = self.builder.build_optimizer(self.config, params=params, total_steps=total_steps)

    def calc_steps(self, dataloader, is_train):
        if is_train:            
            batch_size = self.config.train.batch_size
        else:
            batch_size = self.config.evaluation.batch_size

        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)
        return total_step

    def predict(self):
        # load checkpoint
        ckpt = self.cm.latest()
        assert(ckpt is not None)

        self.cm.load(self.model, None, ckpt)
        
        d = self.dataloaders[0]
        dataloader = d['dataloader']

        self.model.train(False)        
        total_step = self.calc_steps(dataloader, False)
        with torch.set_grad_enabled(False):
            all_outputs = []

            tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for step, (inputs, _) in tbar:
                outputs = self.forward_hook(self.model, inputs, None, device=self.device)
                cureent_step = step / total_step
                tbar.set_description(f'inference {cureent_step: .2f} step')
                all_outputs.extend(outputs.cpu().detach().numpy())

            df = pd.DataFrame(all_outputs, columns=["output"])
            return df.to_numpy()