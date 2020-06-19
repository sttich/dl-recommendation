# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:14:46 2016

@author: Balázs Hidasi
"""

import sys
sys.path.append('../..')

import numpy as np
import pandas as pd
import gru4rec
import evaluation

PATH_TO_TRAIN = ''
PATH_TO_TEST = ''

if __name__ == '__main__':
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId':np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId':np.int64})

    gru = gru4rec.GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad',
                          n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2,
                          momentum=0.3, n_sample=128, sample_alpha=0, bpreg=1, constrained_embedding=False,
                          dwell_time=100)
    gru.fit(data)
    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=5, dwell_time=100)
    result1[0, 0] = res[0]
    result1[0, 1] = res[1]
    result1[0, 2] = res[2]
    print('bpr-max-sample45')
    print('HR@5: {}'.format(res[0]))
    print('MRR@5: {}'.format(res[1]))
    print('NDCG@5: {}'.format(res[2]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=10, dwell_time=100)
    result1[1, 0] = res[0]
    result1[1, 1] = res[1]
    result1[1, 2] = res[2]
    print('HR@10: {}'.format(res[0]))
    print('MRR@10: {}'.format(res[1]))
    print('NDCG@10: {}'.format(res[2]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=20, dwell_time=100)
    result1[2, 0] = res[0]
    result1[2, 1] = res[1]
    result1[2, 2] = res[2]
    print('HR@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    print('NDCG@20: {}'.format(res[2]))



    gru = gru4rec.GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad',
                          n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2,
                          momentum=0.3, n_sample=128, sample_alpha=0, bpreg=1, constrained_embedding=False,
                          dwell_time=75)
    gru.fit(data)
    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=5, dwell_time=75)
    result2[0, 0] = res[0]
    result2[0, 1] = res[1]
    result2[0, 2] = res[2]
    print('bpr-max-sample60')
    print('HR@5: {}'.format(res[0]))
    print('MRR@5: {}'.format(res[1]))
    print('NDCG@5: {}'.format(res[2]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=10, dwell_time=75)
    result2[1, 0] = res[0]
    result2[1, 1] = res[1]
    result2[1, 2] = res[2]
    print('HR@10: {}'.format(res[0]))
    print('MRR@10: {}'.format(res[1]))
    print('NDCG@10: {}'.format(res[2]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=20, dwell_time=75)
    result2[2, 0] = res[0]
    result2[2, 1] = res[1]
    result2[2, 2] = res[2]
    print('HR@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    print('NDCG@20: {}'.format(res[2]))




