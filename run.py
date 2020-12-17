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

from sklearn import metrics
from pathlib import Path
from easydict import EasyDict as edict
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from built.trainer import Trainer
from built.builder import Builder

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
def ensemble_sentiment(_run, _config):
    config = edict(_config)

    config.dataset.splits = []
    config.dataset.splits.append(
        {'train': False, 'csv_path': 'tweet/input/tweet-sentiment-extraction/corrected_new_test.csv'})
    config.train.num_epochs = 1
    builder = Builder()
    ensembler = Ensembler(config, builder)
    ensembled_output, targets = ensembler.forward_models()

    df = pd.read_csv('tweet/input/tweet-sentiment-extraction/new_test.csv')
    df = df.dropna().reset_index(drop=True)
    
    sentiment_target = targets['sentiment_target']
    sentiment_target = sentiment_target.cpu().detach().numpy()

    accuracy = metrics.accuracy_score(sentiment_target, ensembled_output)
    print(f'accuracy: {accuracy}')

    df.rename(columns={'sentiment': 'ori_sentiment'}, inplace=True)

    df['ensembled_sentiment'] = ensembled_output

    sentiment = ['neutral', 'positive', 'negative']

    for index, row in df.iterrows():
        df.at[index, 'sentiment'] = sentiment[row.ensembled_sentiment]

    df.to_csv(
        'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment.csv')


def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0]: offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    if selected_text.strip() == "":
        return text
    return selected_text


@ex.command
def ensemble_index_extraction(_run, _config):
    config = edict(_config)

    config.dataset.splits = []
    config.dataset.splits.append(
        {'train': False, 'split': 'test', 'csv_path': 'tweet/input/tweet-sentiment-extraction/corrected_new_test.csv'})
    config.train.num_epochs = 1
    builder = Builder()
    ensembler = Ensembler(config, builder)
    output, targets = ensembler.forward_models()

    start_pred = output[0]
    end_pred = output[1]

    start_idx = targets['start_idx'].cpu().detach().numpy()
    end_idx = targets['end_idx'].cpu().detach().numpy()

    selected_text_pred = []
    for i in range(start_idx.shape[0]):
        selected_text_pred.append(get_selected_text(
            targets['tweet'][i], start_pred[i], end_pred[i], targets['offsets'][i]))

    df = pd.read_csv(
        'tweet/input/tweet-sentiment-extraction/corrected_new_test.csv')
    df = df.dropna().reset_index(drop=True)

    # selected_text_ori = df['selected_text'] #[]
    selected_text_ori = []
    for i in range(start_idx.shape[0]):
        selected_text_ori.append(get_selected_text(
            targets['tweet'][i], start_idx[i], end_idx[i], targets['offsets'][i]))

    diff_cnt = 0
    for i in range(start_idx.shape[0]):
        if len(selected_text_pred[i]) != len(selected_text_ori[i]):
            diff_cnt += 1
            # print(selected_text_pred[i])
            # print(selected_text[i])
    print(f'total diff cnt: {diff_cnt}')

    start_idx_accuracy = metrics.accuracy_score(start_idx, start_pred)
    end_idx_accuracy = metrics.accuracy_score(end_idx, end_pred)
    print(f'accuracy - start:{start_idx_accuracy}, end:{end_idx_accuracy}')
    
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
    print(f'{score}')

    # jaccard_score = 0.0
    # for i in range(len(selected_text_ori)):
    #     jaccard_score += jaccard(
    #         selected_text_ori[i], selected_text_pred[i])

    # print(f'jacaard score(pred): {jaccard_score/len(selected_text_ori)}')



    df.rename(columns={'selected_text': 'ori_selected_text'}, inplace=True)

    df['selected_text'] = selected_text_pred
    df['selected_text_pred'] = selected_text_pred
    df['start_idx'] = start_idx
    df['end_idx'] = end_idx
    df['start_pred'] = start_pred
    df['end_pred'] = end_pred

    df.to_csv(
        'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index.csv')

    
    # df = pd.read_csv(
    #     'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index.csv')
    # df = df.dropna().reset_index(drop=True)

    # selected_text_ori = df['ori_selected_text'].to_numpy()
    # selected_text_pred = df['selected_text_pred'].to_numpy()

    # jaccard_score = 0.0
    # for i in range(len(selected_text_ori)):
    #     jaccard_score += jaccard(
    #         selected_text_ori[i], selected_text_pred[i])

    # print(f'jacaard score(pred)!!: {jaccard_score/len(selected_text)}') 
    
    
    # jaccard_score = 0.0
    # for i in range(len(selected_text_ori)):
    #     jaccard_score += jaccard(
    #         selected_text_ori[i], selected_text[i])
        
    #     if len(selected_text_ori[i].strip()) != len(selected_text[i].strip()):
    #         print(selected_text_ori[i])
    #         print(selected_text[i])
            

    # print(f'jacaard score(pred)!!!!!!: {jaccard_score/len(selected_text)}') 
    
    
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
        {'train': False, 'split': 'test', 'csv_path': 'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index.csv'})
    config.train.num_epochs = 1
    builder = Builder()
    ensembler = Ensembler(config, builder)
    output, targets = ensembler.forward_models()

    start_pred_coverage = output[0]
    end_pred_coverage = output[1]

    selected_text_pred_coverage = []
    for i in range(start_pred_coverage.shape[0]):
        selected_text_pred_coverage.append(get_selected_text(
            targets['tweet'][i], start_pred_coverage[i], end_pred_coverage[i], targets['offsets'][i]))

    df = pd.read_csv(
        'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index.csv')
    df = df.dropna().reset_index(drop=True)

    selected_text_ori = df['ori_selected_text'].to_numpy()
    start_idx = df['start_idx'].to_numpy()
    end_idx = df['end_idx'].to_numpy()
    
    start_pred = df['start_pred'].to_numpy()
    end_pred = df['end_pred'].to_numpy()
    # start_idx = targets['start_idx'].cpu().detach().numpy()
    # end_idx = targets['end_idx'].cpu().detach().numpy()

    # selected_text = []
    # for i in range(start_idx.shape[0]):
    #     selected_text.append(get_selected_text(
    #         targets['tweet'][i], start_idx[i], end_idx[i], targets['offsets'][i]))

    selected_text_pred = df['selected_text'].to_numpy()

    

    start_idx_accuracy = metrics.accuracy_score(start_idx, start_pred_coverage)
    end_idx_accuracy = metrics.accuracy_score(end_idx, end_pred_coverage)

    # diff_cnt = 0
    # for i in range(start_idx.shape[0]):
    #     if len(selected_text_pred_coverage[i]) != len(selected_text_pred[i]):
    #         diff_cnt += 1
    #         print(selected_text_pred_coverage[i])
    #         print(selected_text_pred[i])
    # print(f'total diff cnt: {diff_cnt}')
    # print(f'accuracy - start:{start_idx_accuracy}, end:{end_idx_accuracy}')

    # jaccard_score_coverage = 0.0
    # jaccard_score_pred = 0.0
    # cnt = 0
    # for i in range(len(selected_text)):
    #     if selected_text_pred_coverage[i] == '':
    #         continue
    #     cnt += 1
    #     jaccard_score_coverage += jaccard(
    #         selected_text_ori[i], selected_text_pred_coverage[i])
    #     jaccard_score_pred += jaccard(
    #         selected_text_ori[i], selected_text_pred[i])

    # print(f'jacaard score(pred): {jaccard_score_pred/cnt}')
    # print(
    #     f'jacaard score(coverage): {jaccard_score_coverage/cnt}')
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

    for i in range(len(selected_text_ori)):
        jaccard_score = compute_jaccard_score(
            targets['tweet'][i],
            start_idx[i] + 1,
            end_idx[i] + 1,
            start_pred_coverage[i],
            end_pred_coverage[i],
            targets['offsets'][i])

        jaccard += jaccard_score

    score = jaccard / len(selected_text_ori)
    print(f'coverage: {score}')

    df.rename(columns={'selected_text': 'selected_text_pred'}, inplace=True)

    df['selected_text'] = selected_text_pred_coverage
    df['selected_text_pred_coverage'] = selected_text_pred_coverage

    df['start_pred_coverage'] = start_pred_coverage
    df['end_pred_coverage'] = end_pred_coverage

    df.to_csv(
        'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index_Coverage.csv')


@ex.command
def ensemble_SE_Esc2(_run, _config):
    config = edict(_config)

    config.dataset.splits = []
    config.dataset.splits.append(
        {'train': False, 'split': 'test', 'csv_path': 'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index_Coverage.csv'})
    config.train.num_epochs = 1
    builder = Builder()
    ensembler = Ensembler(config, builder)
    output, targets = ensembler.forward_models()

    start_pred_coverage = output[0]
    end_pred_coverage = output[1]

    selected_text_pred_coverage2 = []
    for i in range(start_pred_coverage.shape[0]):
        selected_text_pred_coverage2.append(get_selected_text(
            targets['tweet'][i], start_pred_coverage[i], end_pred_coverage[i], targets['offsets'][i]))

    df = pd.read_csv(
        'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index_Coverage.csv')
    df = df.dropna().reset_index(drop=True)

    selected_text_ori = df['ori_selected_text']
    selected_text_pred_coverage = df['selected_text']

    start_idx = df['start_idx'].to_numpy()
    end_idx = df['end_idx'].to_numpy()

    start_idx_accuracy = metrics.accuracy_score(start_idx, start_pred_coverage)
    end_idx_accuracy = metrics.accuracy_score(end_idx, end_pred_coverage)

    # diff_cnt = 0
    # for i in range(start_idx.shape[0]):
    #     if len(selected_text_pred_coverage2[i]) != len(selected_text_pred_coverage[i]):
    #         diff_cnt += 1
    #         print(selected_text_ori[i])
    #         print(selected_text_pred_coverage2[i])
    #         print(selected_text_pred_coverage[i])
    # print(f'total diff cnt: {diff_cnt}')
    # print(f'accuracy - start:{start_idx_accuracy}, end:{end_idx_accuracy}')

    # jaccard_score_coverage2 = 0.0
    # jaccard_score_coverage1 = 0.0
    # for i in range(len(selected_text_ori)):
        
    #     jaccard_score_coverage1 += jaccard(
    #         selected_text_ori[i], selected_text_pred_coverage[i])
    #     jaccard_score_coverage2 += jaccard(
    #         selected_text_ori[i], selected_text_pred_coverage2[i])
    jaccard = 0.0
    for i in range(len(selected_text_ori)):
        jaccard_score = compute_jaccard_score(
            targets['tweet'][i],
            start_idx[i] + 1,
            end_idx[i] + 1,
            start_pred_coverage[i],
            end_pred_coverage[i],
            targets['offsets'][i])

        jaccard += jaccard_score

    score = jaccard / len(selected_text_ori)
    print(f'coverage: {score}')

    

    df.rename(columns={'selected_text': 'selected_text_pred'}, inplace=True)

    df['selected_text_pred_coverage2'] = selected_text_pred_coverage

    df['start_pred_coverage'] = start_pred_coverage
    df['end_pred_coverage'] = end_pred_coverage

    df.to_csv(
        'tweet/input/tweet-sentiment-extraction/corrected_new_test_Sentiment_Index_Coverage_2.csv')


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

        tr = Trainer(config, builder)
        tr.run()
        print(f'Training end\n')


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
