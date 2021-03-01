from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import torch
import pandas as pd
import datetime
import numpy as np
import wandb

from sklearn import metrics
from pathlib import Path
from easydict import EasyDict as edict
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from built.trainer import Trainer
from built.builder import Builder
from built.checkpoint_manager import CheckpointManager
from built.ensembler import Ensembler

from tweet.src.correct_dataset import correct_dataset

ex = Experiment('orsum')
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def cfg():
    description = 'Tweet Sentiment Classification'


@ex.main
def main(_run, _config):
    config = edict(_config)
    pprint.PrettyPrinter(indent=2).pprint(config)

@ex.command
def cam_sentiment(_run, _config):
    config = edict(_config)
    
    checkpoint_path = os.path.join(config.train.dir, 'checkpoint')
    
    cm = CheckpointManager(checkpoint_path)
    ckpt = cm.latest()
    builder = Builder()
    run = wandb.init(
        project=f'cam_{config.wandb.project.name}', group=config.wandb.group.name, reinit=True)
    
    tr = Trainer(config, builder, run)
    
    last_epoch, step, last_accuracy = cm.load(tr.model, tr.optimizer, ckpt)
    print(f'{checkpoint_path}:{last_epoch} , {last_accuracy}')

    output, targets = tr.forward()
            
    
    

@ex.command
def ensemble_sentiment(_run, _config):
    config = edict(_config)
    config.dataset.splits = []
    config.dataset.splits.append(
        {'train': config.ensemble_train, 'split': config.ensemble_split, 'csv_path': config.ensemble_csv_input_path})
    config.train.num_epochs = 1

    builder = Builder()
    run = wandb.init(
        project=f'ensemble_{config.wandb.project.name}', group=config.wandb.group.name, reinit=True)
    ensembler = Ensembler(config, builder, run)
    ensembled_output, targets = ensembler.forward_models()

    sentiment_target = targets['sentiment_target']
    sentiment_target = sentiment_target.cpu().detach().numpy()

    accuracy = metrics.accuracy_score(sentiment_target, ensembled_output)
    precision = metrics.precision_score(
        sentiment_target, ensembled_output, average='micro')

    recall = metrics.recall_score(
        sentiment_target, ensembled_output, average='micro')

    f1_score = metrics.f1_score(
        sentiment_target, ensembled_output, average='micro')
    log_dict = {
        'ensemble_accuracy': accuracy,
        'ensemble_precision': precision,
        'ensemble_recall': recall,
        'ensemble_f1_score': f1_score}

    run.log(log_dict)
    print(log_dict)

    df = pd.read_csv(config.ensemble_csv_input_path)
    df = df.dropna().reset_index(drop=True)
    df.rename(columns={'sentiment': 'ori_sentiment'}, inplace=True)
    df['ensembled_sentiment'] = ensembled_output

    sentiment = ['neutral', 'positive', 'negative']
    for index, row in df.iterrows():
        df.at[index, 'sentiment'] = sentiment[row.ensembled_sentiment]

    df.to_csv(config.ensemble_csv_output_path)





@ex.command
def ensemble_index_extraction(_run, _config):
    config = edict(_config)
    config.dataset.splits = []
    config.dataset.splits.append(
        {'train': config.ensemble_train, 'split': config.ensemble_split, 'csv_path': config.ensemble_csv_input_path})
    config.train.num_epochs = 1

    builder = Builder()
    run = wandb.init(
        project=f'ensemble_{config.wandb.project.name}', group=config.wandb.group.name, reinit=True)
    ensembler = Ensembler(config, builder, run)
    output, targets = ensembler.forward_models()

    start_pred = output[0]
    end_pred = output[1]

    start_idx = targets['start_idx'].cpu().detach().numpy()
    end_idx = targets['end_idx'].cpu().detach().numpy()

    selected_text_pred = []
    for i in range(start_idx.shape[0]):
        selected_text_pred.append(get_selected_text(
            targets['tweet'][i], start_pred[i], end_pred[i], targets['offsets'][i]))

    df = pd.read_csv(config.ensemble_csv_input_path)
    df = df.dropna().reset_index(drop=True)

    selected_text_ori = df[config.original_selected_text_column].to_numpy()

    diff_cnt = 0
    for i in range(start_idx.shape[0]):
        if len(selected_text_pred[i]) != len(selected_text_ori[i]):
            diff_cnt += 1
            # print(selected_text_pred[i])
            # print(selected_text[i])
    print(f'total diff cnt: {diff_cnt}')

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

    jaccard = 0.0

    for i in range(len(selected_text_ori)):
        jaccard_score = compute_jaccard_score(
            targets['tweet'][i],
            start_idx[i],
            end_idx[i],
            start_pred[i],
            end_pred[i],
            targets['offsets'][i])

        jaccard += jaccard_score

    score = jaccard / len(selected_text_ori)

    log_dict = {
        'ensemble_score': score,
        'ensemble_start_idx_accuracy': start_idx_accuracy,
        'ensemble_end_idx_accuracy': end_idx_accuracy,
        'ensemble_start_idx_precision': start_idx_precision,
        'ensemble_end_idx_precision': end_idx_precision,
        'ensemble_start_idx_recall': start_idx_recall,
        'ensemble_end_idx_recall': end_idx_recall,
        'ensemble_start_idx_f1_score': start_idx_f1_score,
        'ensemble_end_idx_f1_score': end_idx_f1_score}

    run.log(log_dict)
    print(log_dict)

    df.rename(columns={'selected_text': 'ori_selected_text'}, inplace=True)

    df['selected_text'] = selected_text_pred
    df['selected_text_pred'] = selected_text_pred
    df['start_idx'] = start_idx
    df['end_idx'] = end_idx
    df['start_pred'] = start_pred
    df['end_pred'] = end_pred

    df.to_csv(config.ensemble_csv_output_path)


def get_selected_text(text, start_idx, end_idx, offsets):
    if start_idx > end_idx:
        return text
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    if selected_text.strip() == "":
        return text
    return selected_text

def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = start_logits
    end_pred = end_logits

    try:
        if start_pred > end_pred:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!')
            pred = text
        else:
            pred = get_selected_text(text, start_pred, end_pred, offsets)

        true = get_selected_text(text, start_idx, end_idx, offsets)
    except:
        raise RuntimeError('something wrong here')

    return jaccard(true, pred)


def jaccard(str1, str2):
    # for s in [',', '.', ';', ':']:
    #     str1 = str1.replace(s, "")
    #     str2 = str2.replace(s, "")

    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


@ex.command
def ensemble_SE_Esc(_run, _config):
    config = edict(_config)

    config.dataset.splits = []
    config.dataset.splits.append(
        {'train': config.ensemble_train, 'split': config.ensemble_split, 'csv_path': config.ensemble_csv_input_path})
    config.train.num_epochs = 1
    builder = Builder()
    run = wandb.init(
        project=f'ensemble_{config.wandb.project.name}', group=config.wandb.group.name, reinit=True)
    ensembler = Ensembler(config, builder, run)
    output, targets = ensembler.forward_models()

    start_pred_coverage = output[0]
    end_pred_coverage = output[1]

    selected_text_pred_coverage = []
    for i in range(start_pred_coverage.shape[0]):
        selected_text_pred_coverage.append(get_selected_text(
            targets['tweet'][i], start_pred_coverage[i], end_pred_coverage[i], targets['offsets'][i]))

    df = pd.read_csv(config.ensemble_csv_input_path)
    df = df.dropna().reset_index(drop=True)

    selected_text_ori = df[config.original_selected_text_column].to_numpy()

    # start_idx = df['start_idx'].to_numpy()
    # end_idx = df['end_idx'].to_numpy()
    start_idx = targets['start_idx'].cpu().detach().numpy()
    end_idx = targets['end_idx'].cpu().detach().numpy()

    # start_pred = df['start_pred'].to_numpy()
    # end_pred = df['end_pred'].to_numpy()

    # selected_text_pred = df['selected_text'].to_numpy()

    start_idx_accuracy = metrics.accuracy_score(start_idx, start_pred_coverage)
    end_idx_accuracy = metrics.accuracy_score(end_idx, end_pred_coverage)

    start_idx_precision = metrics.precision_score(
        start_idx, start_pred_coverage, average='micro')
    end_idx_precision = metrics.precision_score(
        end_idx, end_pred_coverage, average='micro')

    start_idx_recall = metrics.recall_score(
        start_idx, start_pred_coverage, average='micro')
    end_idx_recall = metrics.recall_score(
        end_idx, end_pred_coverage, average='micro')

    start_idx_f1_score = metrics.f1_score(
        start_idx, start_pred_coverage, average='micro')
    end_idx_f1_score = metrics.f1_score(
        end_idx, end_pred_coverage, average='micro')

    # jaccard = 0.0

    # for i in range(len(selected_text_ori)):
    #     jaccard_score = compute_jaccard_score(
    #         targets['tweet'][i],
    #         start_idx[i] + 1,
    #         end_idx[i] + 1,
    #         start_pred[i] + 1,
    #         end_pred[i] + 1,
    #         targets['offsets'][i])

    #     jaccard += jaccard_score

    # score = jaccard / len(selected_text_ori)
    # print(f'pred: {score}')

    jaccard = 0.0

    for i in range(len(selected_text_ori)):
        jaccard_score = compute_jaccard_score(
            targets['tweet'][i],
            start_idx[i],
            end_idx[i],
            start_pred_coverage[i],
            end_pred_coverage[i],
            targets['offsets'][i])

        jaccard += jaccard_score

    score = jaccard / len(selected_text_ori)
    print(f'coverage: {score}')

    log_dict = {
        'ensemble_score': score,
        'ensemble_start_idx_accuracy': start_idx_accuracy,
        'ensemble_end_idx_accuracy': end_idx_accuracy,
        'ensemble_start_idx_precision': start_idx_precision,
        'ensemble_end_idx_precision': end_idx_precision,
        'ensemble_start_idx_recall': start_idx_recall,
        'ensemble_end_idx_recall': end_idx_recall,
        'ensemble_start_idx_f1_score': start_idx_f1_score,
        'ensemble_end_idx_f1_score': end_idx_f1_score}

    run.log(log_dict)
    print(log_dict)

    df.rename(columns={'selected_text': 'selected_text_pred'}, inplace=True)

    df['selected_text'] = selected_text_pred_coverage
    df['selected_text_pred_coverage'] = selected_text_pred_coverage

    df['start_pred_coverage'] = start_pred_coverage
    df['end_pred_coverage'] = end_pred_coverage

    df.to_csv(config.ensemble_csv_output_path)


@ex.command
def ensemble_SE_Esc_using_Es(_run, _config):
    config = edict(_config)

    config.dataset.splits = []
    config.dataset.splits.append(
        {'train': config.ensemble_train, 'split': config.ensemble_split, 'csv_path': config.ensemble_csv_input_path})
    config.train.num_epochs = 1
    builder = Builder()
    run = wandb.init(
        project=f'ensemble_{config.wandb.project.name}', group=config.wandb.group.name, reinit=True)
    ensembler = Ensembler(config, builder, run)
    output, targets = ensembler.forward_models()

    start_pred_coverage = output[0]
    end_pred_coverage = output[1]

    selected_text_pred_coverage = []
    for i in range(start_pred_coverage.shape[0]):
        selected_text_pred_coverage.append(get_selected_text(
            targets['tweet'][i], start_pred_coverage[i], end_pred_coverage[i], targets['offsets'][i]))

    df = pd.read_csv(config.ensemble_csv_input_path)
    df = df.dropna().reset_index(drop=True)

    selected_text_ori = df[config.original_selected_text_column].to_numpy()
    start_idx = df['start_idx'].to_numpy()
    end_idx = df['end_idx'].to_numpy()

    start_pred = df['start_pred'].to_numpy()
    end_pred = df['end_pred'].to_numpy()

    selected_text_pred = df[config.pred_selected_text_column].to_numpy()

    start_idx_accuracy = metrics.accuracy_score(start_idx, start_pred_coverage)
    end_idx_accuracy = metrics.accuracy_score(end_idx, end_pred_coverage)

    start_idx_precision = metrics.precision_score(
        start_idx, start_pred_coverage, average='micro')
    end_idx_precision = metrics.precision_score(
        end_idx, start_pred_coverage, average='micro')

    start_idx_recall = metrics.recall_score(
        start_idx, start_pred_coverage, average='micro')
    end_idx_recall = metrics.recall_score(
        end_idx, start_pred_coverage, average='micro')

    start_idx_f1_score = metrics.f1_score(
        start_idx, start_pred_coverage, average='micro')
    end_idx_f1_score = metrics.f1_score(
        end_idx, start_pred_coverage, average='micro')

    jaccard = 0.0

    for i in range(len(selected_text_ori)):
        jaccard_score = compute_jaccard_score(
            targets['tweet'][i],
            start_idx[i] + 1,
            end_idx[i] + 1,
            start_pred[i] + 1,
            end_pred[i] + 1,
            targets['offsets'][i])

        jaccard += jaccard_score

    score = jaccard / len(selected_text_ori)
    print(f'pred: {score}')

    jaccard = 0.0
    cnt1 = 0 
    cnt2 = 0
    for i in range(len(selected_text_ori)):
        text_len = len(targets['tweet'][i])
        pre_len = end_pred[i] - start_pred[i]

        s = 0
        e = 0
        # print(pre_len / text_len)
        if pre_len / text_len > 0.1:
            s = start_pred_coverage[i]
            e = end_pred_coverage[i]
            cnt1 += 1
        else:
            s = start_pred[i] + 1
            e = end_pred[i] + 1
            cnt2 += 1        
        
        jaccard_score = compute_jaccard_score(
            targets['tweet'][i],
            start_idx[i] + 1,
            end_idx[i] + 1,
            s,
            e,
            targets['offsets'][i])

        jaccard += jaccard_score

    score = jaccard / len(selected_text_ori)
    print(f'coverage: {score}')
    print(cnt1, cnt2)
    log_dict = {
        'ensemble_score': score,
        'ensemble_start_idx_accuracy': start_idx_accuracy,
        'ensemble_end_idx_accuracy': end_idx_accuracy,
        'ensemble_start_idx_precision': start_idx_precision,
        'ensemble_end_idx_precision': end_idx_precision,
        'ensemble_start_idx_recall': start_idx_recall,
        'ensemble_end_idx_recall': end_idx_recall,
        'ensemble_start_idx_f1_score': start_idx_f1_score,
        'ensemble_end_idx_f1_score': end_idx_f1_score}

    run.log(log_dict)
    print(log_dict)

    df.rename(columns={'selected_text': 'selected_text_pred'}, inplace=True)

    df['selected_text'] = selected_text_pred_coverage
    df['selected_text_pred_coverage'] = selected_text_pred_coverage

    df['start_pred_coverage'] = start_pred_coverage
    df['end_pred_coverage'] = end_pred_coverage

    df.to_csv(config.ensemble_csv_output_path)


# @ex.command
# def ensemble_SE_Esc2(_run, _config):
#     config = edict(_config)

#     config.dataset.splits = []
#     config.dataset.splits.append(
#         {'train': False, 'split': 'test', 'csv_path': 'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index_Coverage.csv'})
#     config.train.num_epochs = 1
#     builder = Builder()
#     ensembler = Ensembler(config, builder)
#     output, targets = ensembler.forward_models()

#     start_pred_coverage = output[0]
#     end_pred_coverage = output[1]

#     selected_text_pred_coverage2 = []
#     for i in range(start_pred_coverage.shape[0]):
#         selected_text_pred_coverage2.append(get_selected_text(
#             targets['tweet'][i], start_pred_coverage[i], end_pred_coverage[i], targets['offsets'][i]))

#     df = pd.read_csv(
#         'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index_Coverage.csv')
#     df = df.dropna().reset_index(drop=True)

#     selected_text_ori = df['ori_selected_text']
#     selected_text_pred_coverage = df['selected_text']

#     start_idx = df['start_idx'].to_numpy()
#     end_idx = df['end_idx'].to_numpy()

#     start_idx_accuracy = metrics.accuracy_score(start_idx, start_pred_coverage)
#     end_idx_accuracy = metrics.accuracy_score(end_idx, end_pred_coverage)

#     jaccard = 0.0
#     for i in range(len(selected_text_ori)):
#         jaccard_score = compute_jaccard_score(
#             targets['tweet'][i],
#             start_idx[i] + 1,
#             end_idx[i] + 1,
#             start_pred_coverage[i],
#             end_pred_coverage[i],
#             targets['offsets'][i])

#         jaccard += jaccard_score

#     score = jaccard / len(selected_text_ori)
#     print(f'coverage: {score}')

#     df.rename(columns={'selected_text': 'selected_text_pred'}, inplace=True)

#     df['selected_text_pred_coverage2'] = selected_text_pred_coverage

#     df['start_pred_coverage'] = start_pred_coverage
#     df['end_pred_coverage'] = end_pred_coverage

#     df.to_csv(
#         'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index_Coverage_2.csv')


@ex.command
def train(_run, _config):
    config = edict(_config)
    pprint.PrettyPrinter(indent=2).pprint(config)

    if 'use_date' in config and config['use_date'] is True:
        now = datetime.datetime.now()
        now = now.strftime("%Y%m%d-%H%M%S")
        config.train.dir = os.path.join(config.train.dir, now)

    builder = Builder()
    splitter = builder.build_splitter(config)
    df = pd.read_csv(splitter.csv_path)

    if not os.path.exists(config.train.dir):
        os.makedirs(config.train.dir)

    for i_fold in range(splitter.n_splits):
        run = wandb.init(
            project=config.wandb.project.name, group=config.wandb.group.name, reinit=True)

        print(f'Training start: {i_fold} fold')
        train_idx, val_idx = splitter.get_fold(i_fold)

        train_df = df.iloc[train_idx]
        train_csv_path = os.path.join(
            config.train.dir, str(i_fold) + '_train.csv')
        train_df.to_csv(train_csv_path)

        val_df = df.iloc[val_idx]
        val_csv_path = os.path.join(config.train.dir, str(i_fold) + '_val.csv')
        val_df.to_csv(val_csv_path)
        config.dataset.splits = []
        config.dataset.splits.append(
            {'train': True, 'split': 'train', 'csv_path': train_csv_path})
        config.dataset.splits.append(
            {'train': False, 'split': 'val', 'csv_path': val_csv_path})
        config.dataset.splits.append(
            {'train': False, 'split': 'test', 'csv_path': 'tweet/input/tweet-sentiment-extraction/corrected_new_test.csv'})
        config.train.name = str(i_fold) + '_fold'

        tr = Trainer(config, builder, run)
        tr.run()
        print(f'Training end\n')
        run.finish()


@ex.command
def split(_run, _config):
    config = edict(_config)


    builder = Builder()
    splitter = builder.build_splitter(config)
    df = pd.read_csv(splitter.csv_path)

    output_dir = Path(splitter.csv_path).parent

    fold = 0
    train_idx, val_idx = splitter.get_fold(fold)

    train_df = df.iloc[train_idx]
    train_csv_path = os.path.join(output_dir, 'new_train.csv')
    train_df.to_csv(train_csv_path)
    correct_dataset(train_csv_path)

    val_df = df.iloc[val_idx]
    val_csv_path = os.path.join(output_dir, 'new_test.csv')
    val_df.to_csv(val_csv_path)
    correct_dataset(val_csv_path)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.deterministic = True
    ex.run_commandline()
