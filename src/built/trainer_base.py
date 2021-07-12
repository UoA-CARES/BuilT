import os
import math
import time
import logging
import torch
import tqdm
import numpy as np
import wandb
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from collections import defaultdict


from built.builder import Builder
from built.checkpoint_manager import CheckpointManager
from built.early_stopper import EarlyStopper
from built.logger import LogWriter, WandbWriter
from built.utils.util_functions import *

class TrainerBase(object):
    def __init__(self, config, builder, wandb_run=None, wandb_conf=None, working_dir=None, use_accelerator=False):

        self.config = config

        seed_everything(self.config.train.random_state)

        self.builder = builder
        self.es = EarlyStopper(mode=config.train.early_stopper.mode)        
        self.wandb_run = wandb_run
        self.wandb_conf = wandb_conf
        self.working_dir = working_dir

        if self.working_dir is None:
            self.working_dir = os.path.join(self.config.train.dir, self.config.train.name)
        
        self.cm = CheckpointManager(self.working_dir)

        if self.wandb_run is None:
            self.writer = LogWriter()
        else:
            self.writer = WandbWriter(run=self.wandb_run)

        self.use_accelerator = use_accelerator
        if self.use_accelerator:
            from accelerate import Accelerator

            self.accelerator = Accelerator()
            self.device = self.accelerator.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

        self.build_classes()

    def prepare_directories(self):
        os.makedirs(os.path.join(self.working_dir,
                                 'checkpoint'), exist_ok=True)

    # deprecated, need to check and improve
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

    # def evaluate_single_epoch(self, dataloader, epoch, split):
    #     self.model.eval()

    #     batch_size = self.config.evaluation.batch_size
    #     total_size = len(dataloader.dataset)
    #     total_step = math.ceil(total_size / batch_size)

    #     with torch.no_grad():
    #         all_outputs = []
    #         all_targets = None
    #         aggregated_metric_dict = defaultdict(list)
    #         tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    #         for i, (inputs, targets) in tbar:
    #             output = self.forward_hook(self.model, inputs, targets, device=self.device)
    #             output = self.post_forward_hook(
    #                 outputs=output, inputs=inputs, targets=targets, data=None, is_train=True)

    #             loss = self.loss_fn(output, targets, device=self.device)

    #             if isinstance(loss, dict):
    #                 loss_dict = loss
    #                 loss = loss_dict['loss']
    #             else:
    #                 loss_dict = {'loss': loss}

    #             metric_dict = self.metric_fn(
    #                 outputs=output, targets=targets, data=inputs, is_train=False)                

    #             log_dict = {key: value.item() for key, value in loss_dict.items()}
    #             log_dict['lr'] = self.optimizer.param_groups[0]['lr']
    #             log_dict.update(metric_dict)

    #             for key, value in log_dict.items():
    #                 aggregated_metric_dict[key].append(value)

    #             f_epoch = epoch + i / total_step
    #             tbar.set_description(f'[ val ] {f_epoch: .2f} epoch')
    #             tbar.set_postfix(
    #                 lr=self.optimizer.param_groups[0]['lr'], loss=f'{loss.item():.5f}')
                
    #             if isinstance(output, list) or isinstance(output, tuple):
    #                 for i in range(len(output)):
    #                     if len(all_outputs) < len(output):
    #                         all_outputs.append([])
    #                     all_outputs[i].append(output[i])
    #             else:
    #                 all_outputs.append(output)
                
    #             if isinstance(targets, dict):
    #                 if all_targets is None:
    #                     all_targets = defaultdict(list)
                        
    #                 for k in targets:
    #                     all_targets[k].append(targets[k])            
    #             else:
    #                 if all_targets is None:
    #                     all_targets = []    
    #                 all_targets.append(targets)
                    
    #             self.logger_fn(self.writer, split=split, outputs=output, labels=targets, data=inputs,
    #                                  log_dict=log_dict, epoch=epoch, step=i, num_steps_in_epoch=total_step)
            
    #         aggregated_metric_dict = {f'avg_{key}':np.mean(value) for key, value in aggregated_metric_dict.items()}
    #         self.logger_fn(self.writer, split=split, outputs=all_outputs, labels=all_targets,
    #                                  log_dict=aggregated_metric_dict, epoch=epoch)
    #         return aggregated_metric_dict[f'[{split}]_avg_score']

    # def train_single_epoch(self, dataloader, epoch, split):
    #     self.model.train()

    #     # loop calc
    #     batch_size = self.config.train.batch_size
    #     total_size = len(dataloader.dataset)
    #     total_step = math.ceil(total_size / batch_size)

    #     tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    #     for i, (inputs, targets) in tbar:
    #         output = self.forward_hook(self.model, inputs, targets, device=self.device)
    #         output = self.post_forward_hook(
    #             outputs=output, inputs=inputs, targets=targets, data=None, is_train=True)

    #         loss = self.loss_fn(output, targets, device=self.device)

    #         metric_dict = self.metric_fn(
    #             outputs=output, targets=targets, data=inputs, is_train=True)

    #         if isinstance(loss, dict):
    #             loss_dict = loss
    #             loss = loss_dict['loss']
    #         else:
    #             loss_dict = {'loss': loss}

    #         # backward()
    #         loss.backward()

    #         # optimizer 
    #         if self.config.train.gradient_accumulation_step is None:
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
    #         elif (i+1) % self.config.train.gradient_accumulation_step == 0:
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()

    #         log_dict = {key: value.item() for key, value in loss_dict.items()}
    #         log_dict['lr'] = self.optimizer.param_groups[0]['lr']
    #         log_dict.update(metric_dict)
    #         log_dict.update({'epoch': epoch})

    #         f_epoch = epoch + i / total_step
    #         tbar.set_description(f'[train] {f_epoch: .2f} epoch')
    #         tbar.set_postfix(
    #             lr=self.optimizer.param_groups[0]['lr'], loss=f'{loss.item():.5f}')

    #         self.logger_fn(self.writer, split=split, outputs=output, labels=targets,
    #                              log_dict=log_dict, epoch=epoch, step=i, num_steps_in_epoch=total_step)

    def calc_steps(self, dataloader, is_train):
        if is_train:            
            batch_size = self.config.train.batch_size
        else:
            batch_size = self.config.evaluation.batch_size

        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)
        return total_step

    def process_single_epoch(self, dataloader: DataLoader, epoch: int, is_train: bool, eval_interval: int=1) -> float:
        self.model.train(is_train) 
        # if self.model.training:
        #     print('training mode')
        # else:
        #     print('eval mode')
        # dataloader = self.accelerator.prepare(dataloader)       
        
        total_step = self.calc_steps(dataloader, is_train)
        logger = self.builder.build_logger_fn(self.config, writer=self.writer, epoch=epoch, total_step=total_step, is_train=is_train)
        metric = self.builder.build_metric_fn(self.config)

        with torch.set_grad_enabled(is_train):
            all_outputs = []
            all_targets = None

            tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
            for step, (inputs, targets) in tbar:
                outputs = self.forward_hook(self.model, inputs, targets, device=self.device)
                # outputs = self.post_forward_hook(
                #     outputs=outputs, inputs=inputs, targets=targets, data=None, is_train=True)

                loss = self.loss_fn(outputs, targets, device=self.device)
                if isinstance(loss, dict):
                    loss_dict = loss
                    loss = loss_dict['loss']
                else:
                    loss_dict = {'loss': loss}
                
                lr = self.optimizer.param_groups[0]['lr']

                if not is_train:
                    self.aggregate(all_outputs, outputs, all_targets, targets)

                logger.batch_size = outputs.shape[0]
                logger.log('lr', lr, step)
                logger.log_dict(loss_dict, step)
                logger.log_dict(metric.calculate(outputs=outputs, targets=targets, extra_data=inputs, is_train=is_train), step)
                
                logger.write(step)

                if is_train:
                    self.backward(loss=loss, step=step)

                phase = 'train' if is_train else 'validating'
                tbar.set_postfix(phase=phase, epoch=f'{epoch + 1}', lr=lr, loss=f'{logger.loss:.5f}', score=f'{logger.score:.5f}')

                if is_train and step % eval_interval == 0:
                    print('Validation')
                    score = self.process_single_epoch(self.val_dataloader, epoch, is_train=False)
                    _, save_ckpt = self.es(score)
                    if save_ckpt:
                        self.cm.save(self.model, self.optimizer, epoch+1, score, keep=1, only_state_dict=self.config.train.save_state_dict_only)
                    self.model.train(is_train) 

            return logger.score

    def backward(self, loss, step):
        if self.use_accelerator:
            self.accelerator.backward(loss)
        else:
            loss.backward()

        if self.config.train.gradient_accumulation_step is None:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        elif (step+1) % self.config.train.gradient_accumulation_step == 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        else:
            pass

            
    def aggregate(self, all_outputs, outputs, all_targets, targets):
        if isinstance(outputs, list) or isinstance(outputs, tuple):
            for i in range(len(outputs)):
                if len(all_outputs) < len(outputs):
                    all_outputs.append([])
                all_outputs[i].append(outputs[i])
        else:
            all_outputs.append(outputs)
        
        if isinstance(targets, dict):
            if all_targets is None:
                all_targets = defaultdict(list)
                
            for k in targets:
                all_targets[k].append(targets[k])            
        else:
            if all_targets is None:
                all_targets = []    
            all_targets.append(targets)

    def train(self, last_epoch, last_accuracy=None):
        ckpt_score = last_accuracy

        for epoch in range(last_epoch, self.config.train.num_epochs - 1):
            torch.cuda.synchronize()
            s_time = time.time()

            self.process_single_epoch(self.train_dataloader, epoch, is_train=True)

            torch.cuda.synchronize()
            e_time = time.time() 
            print(f'epoch {epoch} takes {e_time - s_time} seconds.')
        
        return self.es.best_score

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

        # build optimizer
        if 'no_bias_decay' in self.config.train and self.config.train.no_bias_decay:
            param_optimizer = list(self.model.named_parameters())
            no_decay = self.config.optimizer.no_decay
            optimizer_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.optimizer.weight_decay},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0}]
        else:
            optimizer_parameters = self.model.parameters()

        for d in self.dataloaders:
            is_train = d['mode']
            
            if is_train:
                self.train_dataloader = d['dataloader']
            else:
                self.val_dataloader = d['dataloader']
        
        self.total_steps = int(len(self.train_dataloader.dataset) / self.config.train.batch_size * self.config.train.num_epochs)
        self.optimizer = self.builder.build_optimizer(self.config, params=optimizer_parameters, total_steps=self.total_steps)

        if self.use_accelerator:
            self.model, self.optimizer, self.dataloaders[0]['dataloader'], self.dataloaders[1]['dataloader'] = self.accelerator.prepare(
                self.model, self.optimizer, self.dataloaders[0]['dataloader'], self.dataloaders[1]['dataloader'])


    def run(self):        
        last_epoch, step, last_accuracy = -1, -1, None       

        if self.config.train.continue_from_last_checkpoint:
            # load checkpoint
            ckpt = self.cm.latest()
            if ckpt is not None:
                last_epoch, step, last_accuracy = self.cm.load(self.model, self.optimizer, ckpt)

        # build scheduler
        self.scheduler = self.builder.build_scheduler(
            self.config, optimizer=self.optimizer, last_epoch=last_epoch, total_steps=self.total_steps)

        # train loop
        best_score = self.train(last_epoch=last_epoch, last_accuracy=last_accuracy)
        return best_score
