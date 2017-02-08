from  simple_expansion.simple_expansion_feature import get_user_feature, users
from dbscan_clustering import run_DBScan
import cPickle as pickle
from distance_metric import create_metric_matrix
from  feature_engg import create_user_question_matrix

def build_user_data():
    user_features_list = []
    for key in users.keys():
        #print smple_exp.users[key]
        user = users[key]
        user_features_list.append(get_user_feature(user))

    return user_features_list
    
    
def build_similarity_matrix(db):
    
    usersIDVs_clusterId_dict = dict()
    clusterIDVs_userIdList_dict = dict()
    i = 0
    for key in users.keys():
        cluster_id = db.labels_[i]
        usersIDVs_clusterId_dict[key] = cluster_id
        if cluster_id in clusterIDVs_userIdList_dict.keys() and (not clusterIDVs_userIdList_dict[cluster_id] is None):
            clusterIDVs_userIdList_dict[cluster_id].append(key)
        else:
            clusterIDVs_userIdList_dict[cluster_id] = [key]
         
        i += 1   
        
    pickle.dump(clusterIDVs_userIdList_dict, open("clusterIDVs_userIdList_dict.p", "wb"), protocol=2 )
    pickle.dump(usersIDVs_clusterId_dict, open("usersIDVs_clusterId_dict.p", "wb"), protocol=2 )


user_features_list = build_user_data()
#db = run_DBScan(user_features_list)
#build_similarity_matrix(db)
#create_metric_matrix(user_features_list, "user_l1_similarity_metric.p", "l1")
create_metric_matrix(user_features_list, "user_l2_similarity_metric.p", "l2")
create_metric_matrix(create_user_question_matrix.user_to_ques, "user_hamming_similarity_metric.p", "hamming")

