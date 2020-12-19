import os
import math
import logging
import torch
import tqdm
import numpy as np
import wandb

from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

from built.builder import Builder
from built.utils.util_functions import group_weight
from built.checkpoint_manager import CheckpointManager
from built.early_stopper import EarlyStopper


class Trainer(object):
    def __init__(self, config, builder, wandb_run=None, wandb_conf=None):
        self.config = config
        self.builder = builder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.es = EarlyStopper(mode='max')
        
        self.working_dir = os.path.join(self.config.train.dir, self.config.train.name)
        self.cm = CheckpointManager(self.working_dir)
        self.writer = {}
        if config.logger_hook.params.use_tensorboard:
            self.writer['tensorboard'] = SummaryWriter(log_dir=self.working_dir)
        if config.logger_hook.params.use_wandb:
            self.writer['wandb'] = wandb_run

        self.build_classes()

    def prepare_directories(self):
        os.makedirs(os.path.join(self.working_dir,
                                 'checkpoint'), exist_ok=True)

    def forward(self):
        self.model.eval()

        for dataloader in self.dataloaders:
            dataloader = dataloader['dataloader']
            
            batch_size = self.config.evaluation.batch_size
            total_size = len(dataloader.dataset)
            total_step = math.ceil(total_size / batch_size)
            
            all_outputs = []
            all_targets = None
            aggregated_metric_dict = defaultdict(list)
            epoch = 0
            with torch.no_grad():
                tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
                for i, (inputs, targets) in tbar:
                    output = self.forward_hook(self.model, inputs, targets, device=self.device)
                    output = self.post_forward_hook(
                        outputs=output, inputs=inputs, targets=targets, data=None, is_train=True)

                    metric_dict = self.metric_fn(
                        outputs=output, targets=targets, data=inputs, is_train=False)                

                    log_dict = {}
                    log_dict['lr'] = self.optimizer.param_groups[0]['lr']
                    log_dict.update(metric_dict)

                    for key, value in log_dict.items():
                        aggregated_metric_dict[key].append(value)
                    
                    f_epoch = epoch + i / total_step

                    if isinstance(output, list) or isinstance(output, tuple):
                        for i in range(len(output)):
                            if len(all_outputs) < len(output):
                                all_outputs.append([])
                            all_outputs[i].append(output[i])
                    else:
                        all_outputs.append(output)
                    
                    if isinstance(targets, dict):
                        if all_targets is None:
                            all_targets = defaultdict(list)
                            
                        for k in targets:
                            all_targets[k].append(targets[k])            
                    else:
                        if all_targets is None:
                            all_targets = []    
                        all_targets.append(targets)
                        
                    self.logger_fn(self.writer, split='test', outputs=output, labels=targets, data=inputs,
                                        log_dict=log_dict, epoch=epoch, step=i, num_steps_in_epoch=total_step)

                aggregated_metric_dict = {f'avg_{key}':np.mean(value) for key, value in aggregated_metric_dict.items()}
                self.logger_fn(self.writer, split='test', outputs=all_outputs, labels=all_targets,
                                     log_dict=aggregated_metric_dict, epoch=epoch)                                        
                
                if isinstance(all_outputs[0], list):
                    for i in range(len(all_outputs)):
                        all_outputs[i] = torch.cat(all_outputs[i], dim=0)
                else:
                    all_outputs = torch.cat(all_outputs, dim=0)
                    
                if isinstance(all_targets, dict):
                    for k in all_targets:
                        if isinstance(all_targets[k][0], torch.Tensor):
                            all_targets[k] = torch.cat(all_targets[k], dim=0)
                        else:
                            # if it's a list, 
                            tmp = []
                            for v in all_targets[k]:
                                tmp.extend(v)
                            all_targets[k] = tmp
                else:
                    all_targets = torch.cat(all_targets, dim=0)
                    
                return all_outputs, all_targets

    def evaluate_single_epoch(self, dataloader, epoch, split):
        self.model.eval()

        batch_size = self.config.evaluation.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        with torch.no_grad():
            all_outputs = []
            all_targets = None
            aggregated_metric_dict = defaultdict(list)
            tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for i, (inputs, targets) in tbar:
                output = self.forward_hook(self.model, inputs, targets, device=self.device)
                output = self.post_forward_hook(
                    outputs=output, inputs=inputs, targets=targets, data=None, is_train=True)

                loss = self.loss_fn(output, targets, device=self.device)

                if isinstance(loss, dict):
                    loss_dict = loss
                    loss = loss_dict['loss']
                else:
                    loss_dict = {'loss': loss}

                metric_dict = self.metric_fn(
                    outputs=output, targets=targets, data=inputs, is_train=False)                

                log_dict = {key: value.item() for key, value in loss_dict.items()}
                log_dict['lr'] = self.optimizer.param_groups[0]['lr']
                log_dict.update(metric_dict)

                for key, value in log_dict.items():
                    aggregated_metric_dict[key].append(value)

                f_epoch = epoch + i / total_step
                tbar.set_description(f'[ val ] {f_epoch: .2f} epoch')
                tbar.set_postfix(
                    lr=self.optimizer.param_groups[0]['lr'], loss=f'{loss.item():.5f}')
                
                if isinstance(output, list) or isinstance(output, tuple):
                    for i in range(len(output)):
                        if len(all_outputs) < len(output):
                            all_outputs.append([])
                        all_outputs[i].append(output[i])
                else:
                    all_outputs.append(output)
                
                if isinstance(targets, dict):
                    if all_targets is None:
                        all_targets = defaultdict(list)
                        
                    for k in targets:
                        all_targets[k].append(targets[k])            
                else:
                    if all_targets is None:
                        all_targets = []    
                    all_targets.append(targets)
                    
                self.logger_fn(self.writer, split=split, outputs=output, labels=targets, data=inputs,
                                     log_dict=log_dict, epoch=epoch, step=i, num_steps_in_epoch=total_step)
            
            aggregated_metric_dict = {f'avg_{key}':np.mean(value) for key, value in aggregated_metric_dict.items()}
            self.logger_fn(self.writer, split=split, outputs=all_outputs, labels=all_targets,
                                     log_dict=aggregated_metric_dict, epoch=epoch)
            return aggregated_metric_dict[f'[{split}]_avg_score']

    def train_single_epoch(self, dataloader, epoch, split):
        self.model.train()

        batch_size = self.config.train.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (inputs, targets) in tbar:
            output = self.forward_hook(self.model, inputs, targets, device=self.device)
            output = self.post_forward_hook(
                outputs=output, inputs=inputs, targets=targets, data=None, is_train=True)

            loss = self.loss_fn(output, targets, device=self.device)

            metric_dict = self.metric_fn(
                outputs=output, targets=targets, data=inputs, is_train=True)

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
            tbar.set_description(f'[train] {f_epoch: .2f} epoch')
            tbar.set_postfix(
                lr=self.optimizer.param_groups[0]['lr'], loss=f'{loss.item():.5f}')

            self.logger_fn(self.writer, split=split, outputs=output, labels=targets,
                                 log_dict=log_dict, epoch=epoch, step=i, num_steps_in_epoch=total_step)


    def train(self, last_epoch, last_accuracy=None):
        ckpt_score = last_accuracy

        for epoch in range(last_epoch, self.config.train.num_epochs):
            # train
            for d in self.dataloaders:
                is_train = d['mode']
                split = d['split']
                
                if is_train:
                    dataloader = d['dataloader']
                    self.train_single_epoch(dataloader, epoch, split)

            # validation
            for d in self.dataloaders:
                is_train = d['mode']
                split = d['split']
                
                if not is_train:
                    dataloader = d['dataloader']
                    score = self.evaluate_single_epoch(dataloader, epoch, split)
                    if split == 'test':
                        ckpt_score = score
                    else:
                        print(f'score on test: {score}')

            # update learning rate
            self.scheduler.step()

            stop_early, save_ckpt = self.es(ckpt_score)
            if save_ckpt:
                self.cm.save(self.model, self.optimizer, epoch, ckpt_score, keep=2)
            if stop_early:
                break

    def build_classes(self):
        # prepare directories
        self.prepare_directories()

        # build dataloaders
        self.dataloaders = self.builder.build_dataloaders(self.config)

        # build model
        self.model = self.builder.build_model(self.config)
        self.model = self.model.to(self.device)

        # build loss
        self.loss_fn = self.builder.build_loss_fn(self.config)

        # build hooks
        self.forward_hook = self.builder.build_forward_hook(self.config)
        self.post_forward_hook = self.builder.build_post_forward_hook(self.config)

        # build metric
        self.metric_fn = self.builder.build_metric_fn(self.config)

        # build logger
        self.logger_fn = self.builder.build_logger_fn(self.config)

        # build optimizer
        if 'no_bias_decay' in self.config.train and self.config.train.no_bias_decay:
            group_decay, group_no_decay = group_weight(self.model)
            params = [{'params': group_decay}, {
                'params': group_no_decay, 'weight_decay': 0.0}]
        else:
            params = self.model.parameters()
        self.optimizer = self.builder.build_optimizer(self.config, params=params)

    def run(self):
        
        # load checkpoint
        ckpt = self.cm.latest()
        if ckpt is not None:
            last_epoch, step, last_accuracy = self.cm.load(self.model, self.optimizer, ckpt)
        else:
            last_epoch, step, last_accuracy = -1, -1, None

        # build scheduler
        self.scheduler = self.builder.build_scheduler(
            self.config, optimizer=self.optimizer, last_epoch=last_epoch)

        # train loop
        self.train(last_epoch=last_epoch, last_accuracy=last_accuracy)
