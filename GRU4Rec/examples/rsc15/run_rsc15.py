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
    data = pd.read_csv(PATH_TO_TRAIN, sep='\t',dtype={'ItemId': np.int64})
    valid = pd.read_csv(PATH_TO_TEST, sep='\t',dtype={'ItemId': np.int64})

    ##cross-entropy
    gru = gru4rec.GRU4Rec(loss='cross-entropy', final_act='softmax', hidden_act='tanh', layers=[100], adapt='adagrad',
                          n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0.3, learning_rate=0.1,
                          momentum=0.7, n_sample=2048, sample_alpha=0, bpreg=0, constrained_embedding=False)
    gru.fit(data)
    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=5)
    print('cross-entropy-sample0')
    print('HR@5: {}'.format(res[0]))
    print('MRR@5: {}'.format(res[1]))
    print('MAP@5: {}'.format(res[2]))
    print('NDCG@5: {}'.format(res[3]))
    print('PRECISION@5: {}'.format(res[4]))
    print('F1-SCORE@5: {}'.format(res[5]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=10)
    print('HR@10: {}'.format(res[0]))
    print('MRR@10: {}'.format(res[1]))
    print('MAP@10: {}'.format(res[2]))
    print('NDCG@10: {}'.format(res[3]))
    print('PRECISION@10: {}'.format(res[4]))
    print('F1-SCORE@10: {}'.format(res[5]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=20)
    print('HR@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    print('MAP@20: {}'.format(res[2]))
    print('NDCG@20: {}'.format(res[3]))
    print('PRECISION@20: {}'.format(res[4]))
    print('F1-SCORE@20: {}'.format(res[5]))

    # BPR-max
    gru = gru4rec.GRU4Rec(loss='bpr-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad',
                          n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2,
                          momentum=0.3, n_sample=2048, sample_alpha=0, bpreg=1, constrained_embedding=False)
    gru.fit(data)
    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=5)
    print('bpr-max-sample0')
    print('HR@5: {}'.format(res[0]))
    print('MRR@5: {}'.format(res[1]))
    print('MAP@5: {}'.format(res[2]))
    print('NDCG@5: {}'.format(res[3]))
    print('PRECISION@5: {}'.format(res[4]))
    print('F1-SCORE@5: {}'.format(res[5]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=10)
    print('HR@10: {}'.format(res[0]))
    print('MRR@10: {}'.format(res[1]))
    print('MAP@10: {}'.format(res[2]))
    print('NDCG@10: {}'.format(res[3]))
    print('PRECISION@10: {}'.format(res[4]))
    print('F1-SCORE@10: {}'.format(res[5]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=20)
    print('HR@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    print('MAP@20: {}'.format(res[2]))
    print('NDCG@20: {}'.format(res[3]))
    print('PRECISION@20: {}'.format(res[4]))
    print('F1-SCORE@20: {}'.format(res[5]))

    # BPR
    gru = gru4rec.GRU4Rec(loss='bpr', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad',
                          n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2,
                          momentum=0.3, n_sample=2048, sample_alpha=0, bpreg=1, constrained_embedding=False)
    gru.fit(data)
    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=5)
    print('bpr-sample0')
    print('HR@5: {}'.format(res[0]))
    print('MRR@5: {}'.format(res[1]))
    print('MAP@5: {}'.format(res[2]))
    print('NDCG@5: {}'.format(res[3]))
    print('PRECISION@5: {}'.format(res[4]))
    print('F1-SCORE@5: {}'.format(res[5]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=10)
    print('HR@10: {}'.format(res[0]))
    print('MRR@10: {}'.format(res[1]))
    print('MAP@10: {}'.format(res[2]))
    print('NDCG@10: {}'.format(res[3]))
    print('PRECISION@10: {}'.format(res[4]))
    print('F1-SCORE@10: {}'.format(res[5]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=20)
    print('HR@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    print('MAP@20: {}'.format(res[2]))
    print('NDCG@20: {}'.format(res[3]))
    print('PRECISION@20: {}'.format(res[4]))
    print('F1-SCORE@20: {}'.format(res[5]))

    # top1-max
    gru = gru4rec.GRU4Rec(loss='top1-max', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad',
                          n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2,
                          momentum=0.3, n_sample=2048, sample_alpha=0, bpreg=1, constrained_embedding=False)
    gru.fit(data)
    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=5)
    print('top1-max-sample0')
    print('HR@5: {}'.format(res[0]))
    print('MRR@5: {}'.format(res[1]))
    print('MAP@5: {}'.format(res[2]))
    print('NDCG@5: {}'.format(res[3]))
    print('PRECISION@5: {}'.format(res[4]))
    print('F1-SCORE@5: {}'.format(res[5]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=10)
    print('HR@10: {}'.format(res[0]))
    print('MRR@10: {}'.format(res[1]))
    print('MAP@10: {}'.format(res[2]))
    print('NDCG@10: {}'.format(res[3]))
    print('PRECISION@10: {}'.format(res[4]))
    print('F1-SCORE@10: {}'.format(res[5]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=20)
    print('HR@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    print('MAP@20: {}'.format(res[2]))
    print('NDCG@20: {}'.format(res[3]))
    print('PRECISION@20: {}'.format(res[4]))
    print('F1-SCORE@20: {}'.format(res[5]))

    # top1
    gru = gru4rec.GRU4Rec(loss='top1', final_act='elu-0.5', hidden_act='tanh', layers=[100], adapt='adagrad',
                          n_epochs=10, batch_size=32, dropout_p_embed=0, dropout_p_hidden=0, learning_rate=0.2,
                          momentum=0.3, n_sample=2048, sample_alpha=0, bpreg=1, constrained_embedding=False)
    gru.fit(data)
    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=5)
    print('top1-sample0')
    print('HR@5: {}'.format(res[0]))
    print('MRR@5: {}'.format(res[1]))
    print('MAP@5: {}'.format(res[2]))
    print('NDCG@5: {}'.format(res[3]))
    print('PRECISION@5: {}'.format(res[4]))
    print('F1-SCORE@5: {}'.format(res[5]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=10)
    print('HR@10: {}'.format(res[0]))
    print('MRR@10: {}'.format(res[1]))
    print('MAP@10: {}'.format(res[2]))
    print('NDCG@10: {}'.format(res[3]))
    print('PRECISION@10: {}'.format(res[4]))
    print('F1-SCORE@10: {}'.format(res[5]))

    res = evaluation.evaluate_sessions_batch(gru, valid, cut_off=20)
    print('HR@20: {}'.format(res[0]))
    print('MRR@20: {}'.format(res[1]))
    print('MAP@20: {}'.format(res[2]))
    print('NDCG@20: {}'.format(res[3]))
    print('PRECISION@20: {}'.format(res[4]))
    print('F1-SCORE@20: {}'.format(res[5]))




