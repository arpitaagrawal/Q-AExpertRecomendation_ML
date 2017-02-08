import cPickle as pickle
import numpy as np
from  simple_expansion.simple_expansion_feature import users

cluster_user_dict = pickle.load(open("../clusters/cluster_user_dict.p", "rb") )
users_cluster_dict = pickle.load(open("../clusters/users_cluster_dict.p", "rb") )

def get_total_cluster():
    '''
    Returns total number of user clusters found
    '''
    return len(cluster_user_dict)

def get_user_cluster_vector(user_id):
    '''
    Returns a 0/1 array set for cluster user_id belong to
    '''
    vect = [0] * get_total_cluster()
    cluster_id = users_cluster_dict[user_id] + 1
    vect[cluster_id] = 1
    
    return vect

def get_distance_metric_for_user(U1,U2, metric):
    '''
        Returns a scalar value : distance measure between two users
    '''
    if metric == 'l1':
        user_distance_metric  = pickle.load(open('user_l1_similarity_metric.p', "rb") )
    else:
        user_distance_metric  = pickle.load(open('user_hamming_similarity_metric.p', "rb") )
        
    u1_index = users.keys().index(U1)
    u2_index = users.keys().index(U2)
    
    return user_distance_metric[u1_index][u2_index]

    
if __name__ == '__main__':
#     print len(cluster_user_dict), cluster_user_dict[-1]
#     print len(users_cluster_dict), users_cluster_dict[users_cluster_dict.keys()[0]]
#     print set(users_cluster_dict.values())
    print get_distance_metric_for_user("f37635bcd54355ced64ff62b113bc692", "4588a1df2461674252ff01c63b59171a", "l1")
    print get_distance_metric_for_user("f37635bcd54355ced64ff62b113bc692", "4588a1df2461674252ff01c63b59171a", "hamming")
    
    