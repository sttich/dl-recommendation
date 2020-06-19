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

import model_user
import evaluation_user


PATH_TO_TRAIN = ''
PATH_TO_TEST = ''


class Args():
    is_training = False
    layers = 1
    rnn_size = 100
    n_epochs = 3  # the number of epochs
    batch_size = 50
    dropout_p_hidden = 1
    learning_rate = 0.001
    decay = 0.96
    decay_steps = 1e4
    sigma = 0
    init_as_normal = False
    reset_after_session = True
    session_key = 'SessionId'
    item_key = 'ItemId'
    time_key = 'Time'
    user_key = 'UserId'
    grad_cap = 0
    test_model = 2
    checkpoint_dir = './checkpoint'
    loss = 'bpr-max'  # loss function
    final_act = 'softmax'  # activation function of the final layer
    hidden_act = 'tanh'  # activation function of hidden layers
    n_items = -1  # the number of items
    dwell_time = 0  # Default 0, regardless of the impact of dwell time. When greater than 0, the data is adjusted.
    user = 0  # 0: no user in data, 1: taobao user data
    optimizer = 'adam'  # adagrad, adam, adadelta, rmsprop
    boosting_memory = 3  # default 3; When the item boosting is turned on and the array goes out of bounds, it needs to be modified to a larger integer. It shouldn't be too large.
    bpr_max_lambda = 1.0
    n_samples = 0
    sample_alpha = 0.0


def parseArgs():
    parser = argparse.ArgumentParser(description='GRU4Rec args')
    parser.add_argument('--layer', default=1, type=int)
    parser.add_argument('--size', default=50, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--train', default=0, type=int)
    parser.add_argument('--test', default=4, type=int)
    parser.add_argument('--hidden_act', default='tanh', type=str)
    parser.add_argument('--final_act', default='tanh', type=str)
    parser.add_argument('--loss', default='bpr-max', type=str)
    parser.add_argument('--dropout', default='0.5', type=float)
    parser.add_argument('--user', default=1, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--boosting_memory', default=5, type=int)
    parser.add_argument('--bpr_max_lambda', default=1.0, type=float)
    parser.add_argument('--n_samples', default=0, type=int)
    parser.add_argument('--sample_alpha', default=0.0, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    command_line = parseArgs()
    training_data = pd.read_csv(PATH_TO_TRAIN)
    valid_data = pd.read_csv(PATH_TO_TEST)


    user_dic={}
    userids=list(training_data.UserId.unique())
    for i,j in enumerate(userids):
        user_dic[j]=i
    training_data['UserId']=training_data['UserId'].apply(lambda x: user_dic[x])
    valid_data['UserId'] = valid_data['UserId'].apply(lambda x: user_dic[x])

    item_dic = {}
    itemids = list(training_data.ItemId.unique())
    for i, j in enumerate(itemids):
        item_dic[j] = i
    training_data['ItemId'] = training_data['ItemId'].apply(lambda x: item_dic[x])
    valid_data['ItemId'] = valid_data['ItemId'].apply(lambda x: item_dic[x])

    session_dic = {}
    sessionids = list(training_data.SessionId.unique())+list(valid_data.SessionId.unique())
    sessionids=list(set(sessionids))
    for i, j in enumerate(sessionids):
        session_dic[j] = i
    training_data['SessionId'] = training_data['SessionId'].apply(lambda x: session_dic[x])
    valid_data['SessionId'] = valid_data['SessionId'].apply(lambda x: session_dic[x])



    args = Args()
    args.n_items = len(training_data['ItemId'].unique())
    args.n_users = len(training_data['UserId'].unique())
    args.layers = command_line.layer
    args.rnn_size = command_line.size
    args.n_epochs = command_line.epoch
    args.learning_rate = command_line.lr
    args.is_training = command_line.train
    args.test_model = command_line.test
    args.hidden_act = command_line.hidden_act
    args.final_act = command_line.final_act
    args.loss = command_line.loss
    args.dropout_p_hidden = 1.0 if args.is_training == 0 else command_line.dropout
    args.user_data = command_line.user
    args.optimizer = command_line.optimizer
    args.boosting_memory = command_line.boosting_memory
    args.bpr_max_lambda = command_line.bpr_max_lambda
    args.n_samples = command_line.n_samples
    args.sample_alpha = command_line.sample_alpha

    print('dropout: {}'.format(args.dropout_p_hidden))
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    with tf.Session(config=gpu_config) as sess:
        gru = model_user.GRU4Rec(sess, args)  # initialize the parameters and build the model

        if args.is_training == 1:  # training
            if args.user_data == 0:
                gru.fit(training_data)  # data is csv file
            elif args.user_data == 1:
                gru.fit_user(training_data)

        elif args.is_training == 0:  # evaluation
            if args.user_data ==0:
                res = evaluation_user.evaluate_sessions_batch(gru, training_data, valid_data, cut_off=5)
                print('Recall@5: {}'.format(res[0]))
                print('MRR@5: {}'.format(res[1]))
                print('NDCG@5: {}'.format(res[2]))


                res = evaluation_user.evaluate_sessions_batch(gru, training_data, valid_data, cut_off=10)
                print('Recall@10: {}'.format(res[0]))
                print('MRR@10: {}'.format(res[1]))
                print('NDCG@10: {}'.format(res[2]))

                res = evaluation_user.evaluate_sessions_batch(gru, training_data, valid_data, cut_off=20)
                print('Recall@20: {}'.format(res[0]))
                print('MRR@520: {}'.format(res[1]))
                print('NDCG@20: {}'.format(res[2]))
            else:
                print('user')
                res = evaluation_user.evaluate_sessions_batch_user(gru, training_data, valid_data, cut_off=5)
                print('Recall@5: {}'.format(res[0]))
                print('MRR@5: {}'.format(res[1]))
                print('NDCG@5: {}'.format(res[2]))

                res = evaluation_user.evaluate_sessions_batch_user(gru, training_data, valid_data, cut_off=10)
                print('Recall@10: {}'.format(res[0]))
                print('MRR@10: {}'.format(res[1]))
                print('NDCG@10: {}'.format(res[2]))

                res = evaluation_user.evaluate_sessions_batch_user(gru, training_data, valid_data, cut_off=20)
                print('Recall@20: {}'.format(res[0]))
                print('MRR@20: {}'.format(res[1]))
                print('NDCG@20: {}'.format(res[2]))

