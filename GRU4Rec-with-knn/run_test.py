from algorithms.gru4rec import gru4rec
from algorithms.knn import cknn
#from algorithms.knn import iknn
#from algorithms.baselines import sr
#from algorithms.baselines import ar
#from algorithms.baselines import pop as p
#from algorithms.baselines import rpop as rp
#from algorithms.baselines import remind as r
#from algorithms.baselines import random as ran
from algorithms.hybrid import weighted as wh 
#from algorithms.hybrid import cascading as ch
from evaluation import evaluation as eval
from evaluation import loader as loader
from evaluation.metrics import accuracy as ac
from evaluation.metrics import coverage as cov
from evaluation.metrics import popularity as pop
import numpy as np
if __name__ == '__main__':
    
    '''
    Configuration
    '''
    
    data_path=''
    file_prefix='rsc15'
    limit_train = None #limit in number of rows or None
    limit_test = None #limit in number of rows or None
    density_value = 1 #randomly filter out events (0.0-1.0, 1:keep all)
         
    # create a list of metric classes to be evaluated
    metric = []
    metric.append(ac.HitRate(5))
    metric.append( ac.HitRate(10) )
    metric.append( ac.HitRate(20) )
    metric.append( ac.MRR(5) )
    metric.append( ac.MRR(10) )
    metric.append(ac.MRR(20))
    metric.append( ac.NDCG(5) )
    metric.append(ac.NDCG(10))
    metric.append(ac.NDCG(20))

    

    # create a dict of (textual algorithm description => class) to be evaluated
    algs = {}
    
 
    #weighted hybrid example
    hybrid = wh.WeightedHybrid(
        [cknn.ContextKNN(100, 500, similarity="cosine"), gru4rec.GRU4Rec(layers=[100], n_sample=128, sample_alpha=0)],
        [0.3, 0.7])  # weights
    algs['whybrid-test-10-90'] = hybrid

    train, test = loader.load_data(data_path, file_prefix, rows_train=limit_train, rows_test=limit_test,
                                   density=density_value)
    item_ids = train.ItemId.unique()

    # init metrics
    for m in metric:
        m.init(train)

    # train algorithms
    for k, a in algs.items():
        a.fit(train)

    # result dict
    res = {}

    # evaluation
    for k, a in algs.items():
        res = eval.evaluate_sessions(a, metric, test, train)

    result[0, 0] = res[0]
    result[1, 0] = res[1]
    result[2, 0] = res[2]

    result[0, 1] = res[3]
    result[1, 1] = res[4]
    result[2, 1] = res[5]

    result[0, 2] = res[6]
    result[1, 2] = res[7]
    result[2, 2] = res[8]

    print(result)


