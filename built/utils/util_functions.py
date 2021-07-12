import os
import random
import torch
import torch.nn as nn
import numpy as np

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import GroupNorm, Conv2d, Linear, LayerNorm

# def group_weight(module):
#     group_decay = []
#     group_no_decay = []
#     for m in module.modules():
#         if isinstance(m, nn.Linear):
#             group_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, Conv2d):
#             group_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, _BatchNorm):
#             if m.weight is not None:
#                 group_no_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, GroupNorm):
#             if m.weight is not None:
#                 group_no_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, LayerNorm):
#             if m.weight is not None:
#                 group_no_decay.append(m.weight)
#             if m.bias is not None:
#                 group_no_decay.append(m.bias)
#         elif isinstance(m, nn.Embedding):
#             group_decay.append(m.weight)

#     assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
#     return group_decay, group_no_decay


def seed_everything(seed):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
