# -*- coding: utf-8 -*-
"""
Created on Feb 26 2017
Author: Weiping Song
"""
import os
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import pandas as pd
import numpy as np
import argparse

import model_ca
import evaluation_ca



PATH_TO_TRAIN = ''
PATH_TO_TEST = ''

class Args():
    iis_training = False
    layers = 1
    rnn_size = 100
    n_epochs = 10
    batch_size = 50
    dropout_p_hidden=1.0  ###no drop out
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    grad_cap = 0
    test_model = 0
    checkpoint_dir = './checkpoint'
    loss = 'bpr-max'
    final_act = 'elu-0.5'
    hidden_act = 'tanh'
    n_items = -1
    n_samples=128
    sample_alpha= 0.0
    bpr_max_lambda=1.0



def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    # parser.add_argument('--behavior_size', default=100, type=int)
    parser.add_argument('--category_size', default=20, type=int)
    parser.add_argument('--item_size', default=100, type=int)
    parser.add_argument('--rnn_size', default=100, type=int)
    parser.add_argument('--epoch', default=15, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=0, type=int)
    parser.add_argument('--test', default=14, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='softmax', type=str)
    parser.add_argument('--loss', default='bpr-max', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    parser.add_argument('--n_samples', default=0, type=int)
    parser.add_argument('--sample_alpha', default=0, type=float)
    parser.add_argument('--bpr_max_lambda', default=1.0, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)


    return parser.parse_args()


if __name__ == '__main__':
    command_line = parseArgs()
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t',dtype={'ItemId': np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t',dtype={'ItemId': np.int64})


    args = Args()
    args.n_items = len(data['ItemId'].unique())
    # args.n_behaviors = len(data['action_type'].unique())
    args.n_category=len(data['category_id'].unique())
    args.layers = command_line.layer
    args.category_size = command_line.category_size
    args.item_size = command_line.item_size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.n_samples = command_line.n_samples
    args.sample_alpha = command_line.sample_alpha
    args.bpr_max_lambda = command_line.bpr_max_lambda
    args.optimizer = command_line.optimizer
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    print(args.dropout_p_hidden)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    gpu_config = tf.ConfigProto()
    # gpu_config=tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        gru = model_ca.GRU4Rec(sess, args)
        if args.is_training:
            gru.fit(data)
        else:
            print(args.test_model)
            res = evaluation_ca.evaluate_sessions_batch(gru, data, valid, cut_off=5, batch_size=args.batch_size)
            print('Recall@5: {}'.format(res[0]))
            print('MRR@5: {}'.format(res[1]))
            print('NDCG@5: {}'.format(res[2]))

            res = evaluation_ca.evaluate_sessions_batch(gru, data, valid, cut_off=10, batch_size=args.batch_size)
            print('Recall@10: {}'.format(res[0]))
            print('MRR@10: {}'.format(res[1]))
            print('NDCG@10: {}'.format(res[2]))

            res = evaluation_ca.evaluate_sessions_batch(gru, data, valid, cut_off=20, batch_size=args.batch_size)
            print('Recall@20: {}'.format(res[0]))
            print('MRR@20: {}'.format(res[1]))
            print('NDCG@20: {}'.format(res[2]))