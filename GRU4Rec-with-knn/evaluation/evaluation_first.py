import time
import numpy as np

def evaluate_sessions(pr, metrics, test_data, train_data, items=None, cut_off=20, first_num=3, session_key='SessionId', item_key='ItemId', time_key='Time' ): 
    '''
    Evaluates only the first 'first_num' views per session for the algorithms wrt. the given metrics. Has no batch evaluation capabilities. 

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    first_num : int
        How many views at the beginning of a session should be evaluated. 
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')
    
    Returns
    --------
    out :  list of tuples
        (metric_name, value)
    
    '''
    
    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    count = 0
    print('START evaluation first of ', actions, ' actions in ', sessions, ' sessions')
    
    sc = time.clock();
    st = time.time();
    
    time_sum = 0
    time_sum_clock = 0
    time_count = 0
    
    for m in metrics:
        m.reset();
    
    test_data.sort_values([session_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()
    prev_iid, prev_sid = -1, -1
    first_count = 0;
    
    for i in range(len(test_data)):
        
        if count % 1000 == 0:
            print( '    eval process: ', count, ' of ', actions, ' actions: ', ( count / actions * 100.0 ), ' % in',(time.time()-st), 's')
        
        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        if prev_sid != sid:
            prev_sid = sid
            first_count = 0;
        else:
            
            if first_count < first_num:
            
                if items is not None:
                    if np.in1d(iid, items): items_to_predict = items
                    else: items_to_predict = np.hstack(([iid], items))  
                        
                crs = time.clock();
                trs = time.time();
                                
                preds = pr.predict_next(sid, prev_iid, items_to_predict)
                
                preds[np.isnan(preds)] = 0
    #             preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
                preds.sort_values( ascending=False, inplace=True )
                
                time_sum_clock += time.clock()-crs
                time_sum += time.time()-trs
                time_count += 1
                
                for m in metrics:
                    m.add( preds, iid )
                
                first_count += 1
            
            else:
            
                preds = pr.predict_next(sid, prev_iid, items_to_predict, skip=True)
            
        prev_iid = iid
        
        count += 1
    
    print( 'END evaluation first in ', (time.clock()-sc), 'c / ', (time.time()-st), 's' )
    print( '    avg rt ', (time_sum/time_count), 's / ', (time_sum_clock/time_count), 'c' )

    res = []
    for m in metrics:
        res.append( m.result() )
    
    return res
