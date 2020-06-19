# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:20:12 2015

@author: Balázs Hidasi
"""

import numpy as np
import pandas as pd
import datetime as dt

PATH_TO_ORIGINAL_DATA = ''
PATH_TO_PROCESSED_DATA = ''



def data_split():
    data = pd.read_csv(PATH_TO_ORIGINAL_DATA + 'train.csv')
    #print(data.head())
    data.rename(columns={'user_id':'UserId','session_id':'SessionId','timestamp':'Time','reference':'ItemId'},inplace=True)
    select_value=['clickout item','interaction item rating','interaction item info','interaction item image','interaction item deals']
    data=data[np.in1d(data.action_type,select_value)]
    #print(data.columns)
    data = data[['UserId', 'SessionId', 'ItemId', 'Time']]
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths > 1].index)]
    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports >= 5].index)]
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths >= 2].index)]
    user_lengths=data.groupby('UserId')['SessionId'].nunique()
    data = data[np.in1d(data.UserId, user_lengths[user_lengths>10].index)]

    sessions=data.sort_values(by=['UserId','Time']).groupby('UserId')['SessionId']
    session_test=sessions.last()
    test=data[np.in1d(data.SessionId,session_test.values)]
    train=data[~data.SessionId.isin(session_test.values)].copy()

    test = test[np.in1d(test.ItemId, train.ItemId)]
    #test = test[np.in1d(test.UserId, train.UserId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(PATH_TO_PROCESSED_DATA + 'recsys19_train_full.csv', index=False)
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(PATH_TO_PROCESSED_DATA + 'recsys19_test.csv', index=False)

data_split()
