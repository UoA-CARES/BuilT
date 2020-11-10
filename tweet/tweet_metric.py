from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import logging

import numpy as np
import torch

from built.metric import MetricBase

from built.registry import Registry


@Registry.register(category='hooks')
class TweetMetric(MetricBase):
    def __call__(self, outputs, targets, is_train, split):
        logging.debug("Default metric is called")
        outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()
        outputs = np.argmax(outputs, axis=1)
        targets = targets.cpu().detach().numpy()

        # accuracy = metrics.accuracy_score(targets, outputs)
        # accuracy_score.update(accuracy.item(), ids.size(0))
        # losses.update(loss.item(), ids.size(0))

        #return {'score': accuracy, 'accuracy': accuracy}
        return {'score': 0, 'accuracy': 0}
