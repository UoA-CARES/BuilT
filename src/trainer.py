import os
import math
import logging
import torch
import tqdm
import numpy as np
import wandb

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

from .builder import Builder
from .utils import group_weight
from .checkpoint_manager import CheckpointManager
from .early_stopper import EarlyStopper


class Trainer(object):
    def __init__(self, config, wandb_conf=None):
        self.config = config        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.es = EarlyStopper(mode='max')
        self.cm = CheckpointManager(self.config.train.dir)
        self.writer = {}
        if config.logger_hook.params.use_tensorboard:
            self.writer['tensorboard'] = SummaryWriter(log_dir=config.train.dir)
        if config.logger_hook.params.use_wandb:
            self.writer['wandb'] = wandb.init(
                config=wandb_conf, project="BuilT")

    def prepare_directories(self):
        os.makedirs(os.path.join(self.config.train.dir,
                                 'checkpoint'), exist_ok=True)

    def evaluate_single_epoch(self, dataloader, epoch):
        self.model.eval()

        batch_size = self.config.evaluation.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        with torch.no_grad():
            aggregated_metric_dict = defaultdict(list)
            tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
            for i, (data, target) in tbar:
                images = data.to(self.device)
                labels = target.to(self.device)

                output = self.model(images)
                output = self.post_forward_hook(
                    outputs=output, images=images, labels=labels, data=None, is_train=False)

                loss = self.loss_fn(output, labels)

                if isinstance(loss, dict):
                    loss_dict = loss
                    loss = loss_dict['loss']
                else:
                    loss_dict = {'loss': loss}

                metric_dict = self.metric_fn(
                    outputs=output, labels=labels, is_train=True, split=None)                

                log_dict = {key: value.item() for key, value in loss_dict.items()}
                log_dict['lr'] = self.optimizer.param_groups[0]['lr']
                log_dict.update(metric_dict)

                for key, value in log_dict.items():
                    aggregated_metric_dict[key].append(value)

                f_epoch = epoch + i / total_step
                tbar.set_description(f'{f_epoch:.2f} epoch')
                tbar.set_postfix(
                    lr=self.optimizer.param_groups[0]['lr'], loss=loss.item())
                
                self.logger_fn(self.writer, split='test', outputs=output, labels=labels,
                                     log_dict=log_dict, epoch=epoch, step=i, num_steps_in_epoch=total_step)
            
            aggregated_metric_dict = {f'avg_{key}':np.mean(value) for key, value in aggregated_metric_dict.items()}
            self.logger_fn(self.writer, split='test', outputs=None, labels=None,
                                     log_dict=aggregated_metric_dict, epoch=epoch)
            return aggregated_metric_dict['avg_score']

    def train_single_epoch(self, dataloader, epoch):
        self.model.train()

        batch_size = self.config.train.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, (data, target) in tbar:
            images = data.to(self.device)
            labels = target.to(self.device)

            output = self.model(images)
            output = self.post_forward_hook(
                outputs=output, images=images, labels=labels, data=None, is_train=True)

            loss = self.loss_fn(output, labels)

            metric_dict = self.metric_fn(
                outputs=output, labels=labels, is_train=True, split=None)

            if isinstance(loss, dict):
                loss_dict = loss
                loss = loss_dict['loss']
            else:
                loss_dict = {'loss': loss}

            loss.backward()

            if self.config.train.gradient_accumulation_step is None:
                self.optimizer.step()
                self.optimizer.zero_grad()
            elif (i+1) % self.config.train.gradient_accumulation_step == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            log_dict = {key: value.item() for key, value in loss_dict.items()}
            log_dict['lr'] = self.optimizer.param_groups[0]['lr']
            log_dict.update(metric_dict)
            log_dict.update({'epoch': epoch})

            f_epoch = epoch + i / total_step
            tbar.set_description(f'{f_epoch:.2f} epoch')
            tbar.set_postfix(
                lr=self.optimizer.param_groups[0]['lr'], loss=loss.item())

            self.logger_fn(self.writer, split='train', outputs=output, labels=labels,
                                 log_dict=log_dict, epoch=epoch, step=i, num_steps_in_epoch=total_step)


    def train(self, last_epoch):

        for epoch in range(last_epoch, self.config.train.num_epochs):
            # train
            for dataloader in self.dataloaders:
                # split = dataloader['split']
                is_train = dataloader['mode']

                if is_train:
                    dataloader = dataloader['dataloader']
                    self.train_single_epoch(dataloader, epoch)

            # validation
            # score_dict = {}
            ckpt_score = None
            for dataloader in self.dataloaders:
                # split = dataloader['split']
                is_train = dataloader['mode']

                if not is_train:
                    dataloader = dataloader['dataloader']
                    score = self.evaluate_single_epoch(dataloader, epoch)
                    # score_dict[split] = score
                    # Use score of the first split
                    # if ckpt_score is None:
                    ckpt_score = score

            # update learning rate
            self.scheduler.step()

            stop_early, save_ckpt = self.es(ckpt_score)
            if save_ckpt:
                self.cm.save(self.model, self.optimizer, epoch, keep=2)
            if stop_early:
                break

    def run(self):
        # prepare directories
        self.prepare_directories()

        builder = Builder()
        # build dataloaders
        self.dataloaders = builder.build_dataloaders(self.config)

        # build model
        self.model = builder.build_model(self.config)
        self.model = self.model.to(self.device)

        # build loss
        self.loss_fn = builder.build_loss_fn(self.config)

        # build hooks
        self.post_forward_hook = builder.build_post_forward_hook(self.config)

        # build metric
        self.metric_fn = builder.build_metric_fn(self.config)

        # build logger
        self.logger_fn = builder.build_logger_fn(self.config)

        # build optimizer
        if 'no_bias_decay' in self.config.train and self.config.train.no_bias_decay:
            group_decay, group_no_decay = group_weight(self.model)
            params = [{'params': group_decay}, {
                'params': group_no_decay, 'weight_decay': 0.0}]
        else:
            params = self.model.parameters()
        self.optimizer = builder.build_optimizer(self.config, params=params)

        # load checkpoint
        ckpt = self.cm.latest()
        if ckpt is not None:
            last_epoch, step = self.cm.load(self.model, self.optimizer, ckpt)
            print('epoch, step:', last_epoch, step)
        else:
            last_epoch, step = -1, -1

        # build scheduler
        self.scheduler = builder.build_scheduler(
            self.config, optimizer=self.optimizer, last_epoch=last_epoch)

        # train loop
        self.train(last_epoch=last_epoch)
