import cPickle as pickle
import numpy as np
from  simple_expansion.simple_expansion_feature import users
from manage_idx import users_all_idx,user_train_idx


"""

    For this file to execute properly, copy all the user distance metric files (8 files for each metric in total)
    -user_hamming_similarity_metric*.p,  user_l2_similarity_metric*.p to the folder var specified path
    and call get_distance_metric_for_user() method as directed in the init method below.
    
"""
folder = "../clusters/dist_data/"
 
no_of_splits= 8

fname= folder + "user_hamming_similarity_metric"
user_hamming_distance_metric  = pickle.load(open(fname+"_0.p", "rb") )
for i in range(1, no_of_splits):
    user_hamming_distance_metric  = np.vstack((user_hamming_distance_metric, pickle.load(open(fname+"_"+str(i)+".p", "rb") )))
print 'size of hamming distance metric', np.shape(user_hamming_distance_metric) 

 
fname= folder + "user_l2_similarity_metric"
user_l2_distance_metric = []
user_l2_distance_metric  = pickle.load(open(fname+"_0.p", "rb") )
for i in range(1, no_of_splits):
    user_l2_distance_metric  = np.vstack((user_l2_distance_metric, pickle.load(open(fname+"_"+str(i)+".p", "rb") )))
print 'size of l2 distance metric', np.shape(user_l2_distance_metric) 


def get_distance_metric_for_user(U1,U2, metric):
    '''
        Returns a scalar value : distance measure between two users
    '''
    
    if metric== "l2":
        return user_l2_distance_metric[users_all_idx[U1]][users_all_idx[U2]]
    else:
        if U1 not in user_train_idx or U2 not in user_train_idx :
            return None 
        return user_hamming_distance_metric[user_train_idx[U1]][user_train_idx[U2]]
        
    
if __name__ == '__main__':

    print get_distance_metric_for_user("f37635bcd54355ced64ff62b113bc692", "4588a1df2461674252ff01c63b59171a", "l2")
    print get_distance_metric_for_user("f37635bcd54355ced64ff62b113bc692", "4588a1df2461674252ff01c63b59171a", "hamming")
    
    