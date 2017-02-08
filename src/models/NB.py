from sklearn import datasets
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import cPickle as pickle

filename = "NBModels/naive_bayes_model_"


def train_NB(train_data, train_label, user_key):
    
    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)
    nb_model = MultinomialNB()
    #y_pred_prob = nb_model.fit(train_data, train_label).predict_proba(train_data)
    nb_model.fit(train_data, train_label)
    dump_model(user_key, nb_model, train_data,train_label)
    
def train_user_based_models():
    print 'training user based Naive based models'
    
    print 'loading user based simple best word feature file'
    user_based_train_data = pickle.load(open("../feature_engg/feature/user_based_best_word_features.p", "rb"))
    
    print 'loading user based simple best word label file'
    user_based_train_data_lables = pickle.load(open("../feature_engg/feature/user_based_labels.p", "rb"))

    print 'Training user NB models'
    user_keys = user_based_train_data.keys()
    count = 0
    for user_key in user_keys:
        train_NB(user_based_train_data[user_key],user_based_train_data_lables[user_key], user_key)
        count = count + 1
    

def train_ques_based_models():
    print 'training user based Naive based models'
    
    print 'loading question based simple best word feature file'
    ques_based_train_data = pickle.load(open("../feature_engg/feature/ques_based_best_word_features.p", "rb"))
    
    print 'loading question based simple best word label file'
    ques_based_train_data_lables = pickle.load(open("../feature_engg/feature/ques_based_labels.p", "rb"))
 
    print 'Training question NB models'
    ques_keys = ques_based_train_data.keys()
    count = 0
    for ques_key in ques_keys:
        train_NB(ques_based_train_data[ques_key],ques_based_train_data_lables[ques_key], ques_key)
        count = count + 1     
        if (count %100) == 0:
            print count
    

def dump_model(file_suffix, nb_model, testdata, testlabel):
    
    updated_file_name = filename + str(file_suffix)+ ".pkl"
    with open(updated_file_name, 'wb') as fid:
        pickle.dump(nb_model, fid)    

       
if __name__ == "__main__":
    #train_user_based_models()
    train_ques_based_models()




###### Sample code, on how NBs are trained #############
#     iris = datasets.load_iris()
# 
#     gnb = GaussianNB()
#     print np.shape(iris.data)
#     y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
#     
#     print("No. of mislabeled points out of a total %d points : %d" %(iris.data.shape[0], (iris.target != y_pred).sum()))
#     
#     
#     clf = MultinomialNB()
#     y_pred = clf.fit(iris.data, iris.target).predict_proba(iris.data)
#     print np.unique(iris.target)
#     print y_pred[0:10], iris.target[0:10]