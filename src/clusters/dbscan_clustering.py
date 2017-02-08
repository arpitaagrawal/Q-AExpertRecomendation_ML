from sklearn import metrics
from sklearn.cluster import DBSCAN
import numpy as np
import timeit

def run_DBScan(features):
    
    start_time = timeit.default_timer()
    metrix_matrix = create_metric_matrix(features, "metrix_matrix.p")
    print 'db_scan start_time::', start_time
    print 'running DB scan '
    db = DBSCAN(eps=0.9, min_samples=2, metric='precomputed').fit(metrix_matrix)
    print 'db_scan time taken to cluterise data::', timeit.default_timer() - start_time
    print 'Number of unique cluster lables', np.unique(db.labels_)
    print 'Cluster labels', db.labels_
    print 'Length of cluster labels', len(db.labels_)
    return db

def create_metric_matrix(feature_matrix, filename):
    target_fname = open(filename, 'w')
    start_time = timeit.default_timer()
    print 'running metric'
    metrix_matrix = metrics.pairwise.pairwise_distances(feature_matrix, Y=None, metric='l1')
    metrix_matrix = np.around(metrix_matrix, decimals=2)
    print 'Time taken to create metric data::', timeit.default_timer() - start_time
    np.savetxt(open(filename, 'w'), metrix_matrix, fmt='%.2f')
    #pickle.dump(metrix_matrix, open(filename, "wb"), protocol=2 )
    print 'metrix_matrix', metrix_matrix
    return metrix_matrix