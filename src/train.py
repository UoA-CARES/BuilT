import os

from registry import registry as r
from utils import group_weight
from checkpoint_manager import CheckpointManager
from early_stopper import EarlyStopper


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config.train.batch_size

    def prepare_directories(self):
        os.makedirs(os.path.join(self.config.train.dir,
                                 'checkpoint'), exist_ok=True)

    def train_single_epoch(self, model, split, dataloader, hooks, optimizer, scheduler, epoch):
        model.train()

        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / self.batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
        for i, data in tbar:
            images = data['image'].cuda()
            labels = data['label'].cuda()

            outputs = hooks.forward_fn(
                model=model, images=images, labels=labels, data=data, is_train=True)
            outputs = hooks.post_forward_fn(
                outputs=outputs, images=images, labels=labels, data=data, is_train=True)
            loss = hooks.loss_fn(
                outputs=outputs, labels=labels.float(), data=data, is_train=True)
            metric_dict = hooks.metric_fn(
                outputs=outputs, labels=labels, data=data, is_train=True, split=split)

            if isinstance(loss, dict):
                loss_dict = loss
                loss = loss_dict['loss']
            else:
                loss_dict = {'loss': loss}

            loss.backward()

            if self.config.train.gradient_accumulation_step is None:
                optimizer.step()
                optimizer.zero_grad()
            elif (i+1) % self.config.train.gradient_accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            log_dict = {key: value.item() for key, value in loss_dict.items()}
            log_dict['lr'] = optimizer.param_groups[0]['lr']
            log_dict.update(metric_dict)

            f_epoch = epoch + i / total_step
            tbar.set_description(f'{split}, {f_epoch:.2f} epoch')
            tbar.set_postfix(
                lr=optimizer.param_groups[0]['lr'], loss=loss.item())

            hooks.logger_fn(split=split, outputs=outputs, labels=labels,
                            log_dict=log_dict, epoch=epoch, step=i, num_steps_in_epoch=total_step)

    def train(self, model, loss, optimizer, scheduler, dataloaders, hooks, last_epoch):
        es = EarlyStopper(mode='max')
        cm = CheckpointManager(self.config.train.dir)

        for epoch in range(last_epoch, self.config.train.num_epochs):
            # train
            for dataloader in dataloaders:
                split = dataloader['split']
                dataset_mode = dataloader['mode']

                if dataset_mode == 'train':
                    dataloader = dataloader['dataloader']
                    self.train_single_epoch(
                        model, split, dataloader, optimizer, scheduler, epoch)

            # validation
            score_dict = {}
            ckpt_score = None
            for dataloader in dataloaders:
                split = dataloader['split']
                dataset_mode = dataloader['mode']

                if dataset_mode == 'validation':
                    dataloader = dataloader['dataloader']
                    score = self.evaluate_single_epoch(
                        model, split, dataloader, epoch)
                    score_dict[split] = score
                    # Use score of the first split
                    if ckpt_score is None:
                        ckpt_score = score

            # update learning rate
            scheduler.step()

            stop_early, save_ckpt = es(ckpt_score)
            if save_ckpt:
                cm.save(model, optimizer, epoch, keep=2)
            if stop_early:
                break

    def run(self, config):
        # prepare directories
        self.prepare_directories(config)

        # build model
        model = r.build_model(config)
        model = model.cuda()

        # build loss
        loss = r.build_loss(config)

        # build optimizer
        if 'no_bias_decay' in config.train and config.train.no_bias_decay:
            group_decay, group_no_decay = group_weight(model)
            params = [{'params': group_decay}, {
                'params': group_no_decay, 'weight_decay': 0.0}]
        else:
            params = model.parameters()
        optimizer = r.build_optimizer(config, params=params)

        # build scheduler
        scheduler = r.build_scheduler(config, optimizer=optimizer)

        # train loop
        self.train(config=config, model=model, loss=loss, optimizer=optimizer,
                   scheduler=scheduler, dataloaders=None, hooks=None, last_epoch=None)
