from  simple_expansion.simple_expansion_feature import get_ques_feature, questions
from dbscan_clustering import run_DBScan
import cPickle as pickle
from distance_metric import create_metric_matrix
from  feature_engg import create_user_question_matrix
import numpy as np
def build_ques_data():
    ques_features_list = []
    for key in questions.keys():
        #print smple_exp.users[key]
        question = questions[key]
        ques_features_list.append(get_ques_feature(question))

    return ques_features_list

def build_ques_cluster_dict(db):
    
    questionIDVs_clusterID_dict = dict()
    clusterIDVs_questionId_dict = dict()
    i = 0
    for key in questions.keys():
        cluster_id = db.labels_[i]
        questionIDVs_clusterID_dict[key] = cluster_id
        if cluster_id in clusterIDVs_questionId_dict.keys() and (not clusterIDVs_questionId_dict[cluster_id] is None):
            clusterIDVs_questionId_dict[cluster_id].append(key)
        else:
            clusterIDVs_questionId_dict[cluster_id] = [key]
         
        i += 1   

    pickle.dump(clusterIDVs_questionId_dict, open("cluster_ques_dict.p", "wb"), protocol=2 )
    pickle.dump(questionIDVs_clusterID_dict, open("ques_cluster_dict.p", "wb"), protocol=2 )

print 'bilding data'
ques_features_list = build_ques_data()
#print ques_features_list
#print 'running DB scan & metric'
#db = run_DBScan(ques_features_list)
#build_ques_cluster_dict(db)
#create_metric_matrix(ques_features_list, "ques_l1_similarity_metric.p", "l1")
create_metric_matrix(ques_features_list, "ques_l2_similarity_metric.p", "l2")

create_metric_matrix(np.transpose(create_user_question_matrix.user_to_ques), "ques_hamming_similarity_metric.p", "hamming")