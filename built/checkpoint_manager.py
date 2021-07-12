from __future__ import print_function

import os
import shutil
import torch


class CheckpointManager:
    def __init__(self, train_dir, prefix='epoch_', ext='.pth'):
        assert(train_dir is not None)
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        self.root_dir = os.path.join(train_dir, 'checkpoint')
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        self.prefix = prefix
        self.ext = ext

    def list_cp(self, sort=False):
        checkpoints = [checkpoint
                       for checkpoint in os.listdir(self.root_dir)
                       if checkpoint.startswith(self.prefix) and checkpoint.endswith(self.ext)]
        if(sort):
            checkpoints = sorted(checkpoints)
        return checkpoints

    def latest(self):
        checkpoints = self.list_cp()
        if checkpoints:
            return os.path.join(self.root_dir, list(sorted(checkpoints))[-1])
        return None

    def keep_last_n(self, n):
        checkpoints = self.list_cp(sort=True)
        for checkpoint in checkpoints[:-n]:
            os.remove(os.path.join(self.root_dir, checkpoint))

    def copy_last_n(self, n, name):
        checkpoints = self.list_cp(sort=True)
        for i, checkpoint in enumerate(checkpoints[-n:]):
            shutil.copyfile(os.path.join(self.root_dir, checkpoint),
                            os.path.join(self.root_dir, name.format(i)))

    def load(self, model, optimizer, checkpoint, only_state_dict=True):
        print('load checkpoint from', checkpoint)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if only_state_dict:
            return 
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])

        step = checkpoint['step'] if 'step' in checkpoint else -1
        last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1
        last_accuracy = checkpoint['accuracy'] if 'epoch' in checkpoint else -1

        return last_epoch, step, last_accuracy

    def save(self, model, optimizer, epoch, accuracy, step=0, keep=None, only_state_dict=True):
        checkpoint_path = os.path.join(
            self.root_dir, f'{self.prefix}{epoch:04d}{self.ext}')

        state_dict = {}
        for key, value in model.state_dict().items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            state_dict[key] = value

        if only_state_dict:
            weights_dict = {'state_dict': state_dict}
        else:
            weights_dict = {
                'state_dict': state_dict,
                'optimizer_dict': optimizer.state_dict(),
                'epoch': epoch,
                'step': step,
                'accuracy': accuracy,
            }

        torch.save(weights_dict, checkpoint_path)

        if keep is not None and keep > 0:
            self.keep_last_n(keep)
