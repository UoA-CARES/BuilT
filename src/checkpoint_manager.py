from __future__ import print_function

import os
import shutil
import torch


class CheckpointManager:
    def __init__(self, config, prefix='epoch_', ext='.pth'):
        self.root_dir = os.path.join(config.train.dir, 'checkpoint')
        self.prefix = prefix
        self.ext = ext

    def list_cp(self, sort=False):
        checkpoints = [checkpoint
                       for checkpoint in os.listdir(self.root_dir)
                       if checkpoint.startswith(self.prefix) and checkpoint.endswith(ext)]
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

    def load(self, model, optimizer, checkpoint):
        print('load checkpoint from', checkpoint)
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_dict'])

        step = checkpoint['step'] if 'step' in checkpoint else -1
        last_epoch = checkpoint['epoch'] if 'epoch' in checkpoint else -1

        return last_epoch, step

    def save(self, model, optimizer, epoch, step=0, keep=None):
        checkpoint_path = os.path.join(
            self.root_dir, f'{self.prefix}{epoch:04d}{self.ext}')

        state_dict = {}
        for key, value in model.state_dict().items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            state_dict[key] = value

        weights_dict = {
            'state_dict': state_dict,
            'optimizer_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
        }
        torch.save(weights_dict, checkpoint_path)

        if keep is not None and keep > 0:
            self.keep_last_n(keep)
