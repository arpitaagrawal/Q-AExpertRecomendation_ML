from sklearn import metrics
import numpy as np
import timeit
import cPickle as pickle

def calculate_distance_metric(data, filename):
    create_metric_matrix(data, filename, 'l1') 
    
"""
 possible values for metric argument : 'cityblock', 'cosine', 'euclidean', 'l1', 'l2',
    'manhattan'
"""
def  create_metric_matrix(feature_matrix, filename, metric):
    
    start_time = timeit.default_timer()
    print 'running metric'
    #metrix_matrix = metrics.pairwise.pairwise_distances(feature_matrix, Y=None, metric='l1')
    metrix_matrix = metrics.pairwise.pairwise_distances(feature_matrix, Y=None, metric=metric)
    print 'metric matrix size', np.shape(metrix_matrix)
    print 'Time taken to create metric data::', timeit.default_timer() - start_time
    pickle.dump(np.around(metrix_matrix, decimals=6), open(filename, "wb"), protocol=2 )
    #np.savetxt(open(filename, 'w'), metrix_matrix, fmt='%.2f')
    #print 'metrix_matrix', metrix_matrix
    return metrix_matrix

