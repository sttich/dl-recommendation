# -*- coding: utf-8 -*-
"""
Created on Feb 27 2017
Author: Weiping Song
"""
import numpy as np
import pandas as pd



def evaluate_sessions_batch(model, train_data, test_data, cut_off=20, batch_size=50,
                            session_key='SessionId',
                            item_key='ItemId', time_key='Time'):
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.
    Parameters
    --------
    model : A trained GRU4Rec model.
    train_data : It contains the transactions of the train set. In evaluation phrase, this is used to build item-to-id map.
    test_data : It contains the transactions of the test set. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 100.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    Returns
    --------
    out : tuple
        (Recall@N, MRR@N)
    '''
    model.predict = False
    # Build itemidmap from train data.
    itemids = train_data[item_key].unique()
    itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)

    test_data.sort_values([session_key, time_key], inplace=True)  # sort testing data by session and time
    test_data = model.item_boosting(test_data)
    # item boosting: the number of occurrence is dwell_time_i / threshold_time

    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    #print(offset_sessions)
    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1
    iters = np.arange(batch_size).astype(np.int32)
    maxiter = iters.max()
    start = offset_sessions[iters]
    end = offset_sessions[iters + 1]
    in_idx = np.zeros(batch_size, dtype=np.int32)
    np.random.seed(42)

    recall = 0.0
    mrr = 0.0
    retrieved = 0
    ndcg=0.0
    evaluation_point_count = 0

    while True:
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        minlen = (end[valid_mask] - start_valid).min()
        in_idx[valid_mask] = test_data[item_key].values[start_valid]
        for i in range(minlen - 1):
            out_idx = test_data[item_key].values[start_valid + i + 1]  # array [batch_size]
            preds = model.predict_next_batch(iters, in_idx, itemidmap,
                                             batch_size)  # predicted scores: DataFrame [n_items, batch_size]
            preds.fillna(0, inplace=True)
            in_idx[valid_mask] = out_idx
            others = preds.values.T[valid_mask].T
            targets = np.diag(preds.ix[in_idx].values)[valid_mask]
            ranks = (others > targets).sum(axis=0) + 1  # the rank of output items: array [batch_size]
            rank_ok = ranks <= cut_off
            recall += rank_ok.sum()
            # mrr += (1.0 / ranks[rank_ok]).sum()
            mrr += ((1.0 / ranks) * (rank_ok)).sum()
            ndcg += ((1.0 / np.log2(ranks+1)) * (rank_ok)).sum()
            retrieved += len(ranks) * cut_off
            evaluation_point_count += len(ranks)



        start = start + minlen - 1
        mask = np.arange(len(iters))[(valid_mask) & (end - start <= 1)]
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions) - 1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter + 1]


    return recall / evaluation_point_count, mrr / evaluation_point_count, ndcg / evaluation_point_count





def evaluate_sessions_batch_user(model, train_data, test_data, cut_off=20, batch_size=50,
                                 session_key='SessionId',
                                 item_key='ItemId', time_key='Time', user_key='UserId', bootstrap_length=-1,
                                 items=None, break_ties=False, output_rankings=False):
    # In case someone would try to run with both items=None and not None on the same model
    # without realizing that the predict function needs to be replaced
    model.predict = None

    # Build itemidmap from train data.
    itemids = train_data[item_key].unique()
    itemidmap = pd.Series(data=np.arange(len(itemids)), index=itemids)

    # Build useridmap from train data.
    userids = train_data[user_key].unique()
    useridmap = pd.Series(data=np.arange(len(userids)), index=userids)

    # use the training sessions of the users in test_data to bootstrap the state of the user RNN
    test_users = test_data[user_key].unique()
    train_data = train_data[train_data[user_key].isin(test_users)].copy()

    # select the bootstrap_length recent sessions in training data to bootstrap the hidden state of the predictor
    if bootstrap_length >= 0:
        user_sessions = train_data.sort_values(by=[user_key, time_key])[[user_key, session_key]].drop_duplicates()
        session_order = user_sessions.groupby(user_key, sort=False).cumcount(ascending=False)
        last_sessions = user_sessions[session_order < bootstrap_length][session_key]
        train_data = train_data[train_data[session_key].isin(last_sessions)].copy()

    # concatenate training and test sessions
    train_data['in_eval'] = False
    test_data['in_eval'] = True
    test_data = pd.concat([train_data, test_data])

    # pre-process the session data
    user_indptr, offset_sessions, test_data = model.init_user_data(test_data)  # including item boosting
    offset_users = offset_sessions[user_indptr]

    # get the other columns in the dataset
    columns = [user_key, session_key, item_key]
    other_columns = test_data.columns.values[np.in1d(test_data.columns.values, columns, invert=True)].tolist()
    other_columns.remove('in_eval')

    recall = 0.0
    mrr = 0.0
    retrieved = 0
    ndcg=0.0
    evaluation_point_count = 0

    # here we use parallel minibatches over users
    if len(offset_users) - 1 < batch_size:
        batch_size = len(offset_users) - 1

    # variables used to iterate over users
    user_iters = np.arange(batch_size).astype(np.int32)
    user_maxiter = user_iters.max()
    user_start = offset_users[user_iters]
    user_end = offset_users[user_iters + 1]

    # variables to manage iterations over sessions
    session_iters = user_indptr[user_iters]
    session_start = offset_sessions[session_iters]
    session_end = offset_sessions[session_iters + 1]

    in_item_id = np.zeros(batch_size, dtype=np.int32)
    in_user_id = np.zeros(batch_size, dtype=np.int32)
    in_session_id = np.zeros(batch_size, dtype=np.int32)

    np.random.seed(42)
    perc = 10
    n_users = len(offset_users)
    user_cnt = 0
    while True:
        # iterate only over the valid entries in the minibatch
        valid_mask = np.logical_and(user_iters >= 0, session_iters >= 0)
        if valid_mask.sum() == 0:
            break

        session_start_valid = session_start[valid_mask]
        session_end_valid = session_end[valid_mask]
        session_minlen = (session_end_valid - session_start_valid).min()
        in_item_id[valid_mask] = test_data[item_key].values[session_start_valid]
        in_user_id[valid_mask] = test_data[user_key].values[session_start_valid]
        in_session_id[valid_mask] = test_data[session_key].values[session_start_valid]

        for i in range(session_minlen - 1):
            out_item_idx = test_data[item_key].values[session_start_valid + i + 1]
            if items is not None:  # not test yet
                uniq_out = np.unique(np.array(out_item_idx, dtype=np.int32))
                preds = model.predict_next_batch_user(in_session_id, in_item_id, in_user_id,
                                                      np.hstack([items, uniq_out[~np.in1d(uniq_out, items)]]),
                                                      batch_size)
            else:
                preds = model.predict_next_batch_user(in_session_id, in_item_id, in_user_id, itemidmap, useridmap,
                                                      batch_size)
            if break_ties:
                preds += np.random.rand(*preds.values.shape) * 1e-8

            preds.fillna(0, inplace=True)

            in_item_id[valid_mask] = out_item_idx
            in_eval_mask = np.zeros(batch_size, dtype=np.bool)
            in_eval_mask[valid_mask] = test_data['in_eval'].values[session_start_valid + i + 1]

            if np.any(in_eval_mask):
                if items is not None:
                    others = preds.ix[items].values.T[in_eval_mask].T
                    targets = np.diag(preds.ix[in_item_id].values)[in_eval_mask]
                    ranks = (others > targets).sum(axis=0) + 1
                else:
                    ranks = (preds.values.T[in_eval_mask].T > np.diag(preds.ix[in_item_id].values)[
                        in_eval_mask]).sum(
                        axis=0) + 1
                if output_rankings:
                    session_start_eval = session_start[in_eval_mask]
                    eval_record = [in_user_id[in_eval_mask],  # user id
                                   in_session_id[in_eval_mask],  # session id
                                   in_item_id[in_eval_mask],  # OUTPUT item id (see line 261)
                                   ranks]
                    others_record = np.vstack(
                        [test_data[c].values[session_start_eval + i + 1] for c in other_columns])
                    batch_results = np.vstack([eval_record, others_record]).T
                    #rank_list.append(batch_results)

                # Calculate recall, mrr
                rank_ok = ranks <= cut_off
                recall += rank_ok.sum()
                # mrr += (1.0 / ranks[rank_ok]).sum()
                mrr += ((1.0 / ranks) * (rank_ok)).sum()
                ndcg += ((1.0 / np.log2(ranks+1)) * (rank_ok)).sum()
                retrieved += len(ranks) * cut_off
                evaluation_point_count += len(ranks)


        session_start[valid_mask] = session_start[valid_mask] + session_minlen - 1
        session_start_mask = np.arange(len(user_iters))[valid_mask & (session_end - session_start <= 1)]
        for idx in session_start_mask:
            session_iters[idx] += 1
            if session_iters[idx] + 1 >= len(offset_sessions):
                session_iters[idx] = -1
                user_iters[idx] = -1
                break
            session_start[idx] = offset_sessions[session_iters[idx]]
            session_end[idx] = offset_sessions[session_iters[idx] + 1]

        user_change_mask = np.arange(len(user_iters))[valid_mask & (user_end - session_start <= 0)]
        for idx in user_change_mask:
            user_cnt += 1
            if user_cnt > int(perc * n_users / 100):
                print('User {}/{} ({}% completed)'.format(user_cnt, n_users, perc))
                perc += 10
            user_maxiter += 1
            if user_maxiter + 1 >= len(offset_users):
                session_iters[idx] = -1
                user_iters[idx] = -1
                break
            user_iters[idx] = user_maxiter
            user_start[idx] = offset_users[user_maxiter]
            user_end[idx] = offset_users[user_maxiter + 1]
            session_iters[idx] = user_indptr[user_maxiter]
            session_start[idx] = offset_sessions[session_iters[idx]]
            session_end[idx] = offset_sessions[session_iters[idx] + 1]

    return recall / evaluation_point_count, mrr / evaluation_point_count, ndcg/ evaluation_point_count


