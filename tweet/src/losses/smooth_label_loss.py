
import torch
import torch.nn as nn

from built.loss import LossBase
from built.registry import Registry
from built.utils.smooth_label_loss import SmoothLabelCritierion


@Registry.register(category="loss")
class SmoothLabelLoss(LossBase):
    def __call__(self, outputs, targets, data=None, is_train=False, device='cpu'):        
        start_preds = outputs[0]
        start_positions = targets['start_idx'].cuda()
        
        end_preds = outputs[1]
        end_positions = targets['end_idx'].cuda()

        loss_fn = SmoothLabelCritierion(label_smoothing=0.15)
        start_loss = loss_fn(start_preds, start_positions)
        end_loss = loss_fn(end_preds, end_positions)
        total_loss = start_loss + end_loss

        return total_loss
