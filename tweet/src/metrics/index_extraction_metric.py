from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc
import logging

import numpy as np
import torch

from sklearn import metrics
from built.metric import MetricBase
from built.registry import Registry


def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    if selected_text.strip() == "":
        return text
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = start_logits
    end_pred = end_logits

    try:
        if start_pred > end_pred:
            pred = text
        else:
            pred = get_selected_text(text, start_pred, end_pred, offsets)

        true = get_selected_text(text, start_idx, end_idx, offsets)
    except:
        raise RuntimeError('something wrong here')

    return jaccard(true, pred)


@Registry.register(category='hooks')
class TweetIndexExtractionMetric(MetricBase):
    def __call__(self, outputs, targets, data=None, is_train=False, device='cpu'):
        start_idx = targets['start_idx'].cpu().detach().numpy()
        end_idx = targets['end_idx'].cpu().detach().numpy()

        start_pred = torch.softmax(
            outputs[0], dim=1).cpu().detach().numpy()
        start_pred = np.argmax(start_pred, axis=1)

        end_pred = torch.softmax(
            outputs[1], dim=1).cpu().detach().numpy()
        end_pred = np.argmax(end_pred, axis=1)

        start_idx_accuracy = metrics.accuracy_score(start_idx, start_pred)
        end_idx_accuracy = metrics.accuracy_score(end_idx, end_pred)

        start_idx_precision = metrics.precision_score(
            start_idx, start_pred, average='micro')
        end_idx_precision = metrics.precision_score(
            end_idx, end_pred, average='micro')

        start_idx_recall = metrics.recall_score(
            start_idx, start_pred, average='micro')
        end_idx_recall = metrics.recall_score(
            end_idx, end_pred, average='micro')

        start_idx_f1_score = metrics.f1_score(
            start_idx, start_pred, average='micro')
        end_idx_f1_score = metrics.f1_score(end_idx, end_pred, average='micro')

        # start_idx_auc = metrics.roc_auc_score(
        #     start_idx, start_pred)
        # end_idx_auc = metrics.roc_auc_score(
        #     end_idx, end_pred)

        ids = data['ids']
        tweet = data['tweet']
        offsets = data['offsets'].cpu().numpy()

        jaccard = 0.0

        for i in range(len(ids)):
            jaccard_score = compute_jaccard_score(
                tweet[i],
                start_idx[i],
                end_idx[i],
                start_pred[i],
                end_pred[i],
                offsets[i])

            jaccard += jaccard_score

        score = jaccard / len(ids)

        return {
            'score': score,
            'start_idx_accuracy': start_idx_accuracy,
            'end_idx_accuracy': end_idx_accuracy,
            'start_idx_precision': start_idx_precision,
            'end_idx_precision': end_idx_precision,
            'start_idx_recall': start_idx_recall,
            'end_idx_recall': end_idx_recall,
            'start_idx_f1_score': start_idx_f1_score,
            'end_idx_f1_score': end_idx_f1_score}
            # 'start_idx_auc': start_idx_auc,
            # 'end_idx_auc': end_idx_auc}
