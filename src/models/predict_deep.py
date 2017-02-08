'''
Uses flattened features in feature directory and run a SVM on it
'''
 
from keras.models import load_model
import numpy as np
import cPickle as pickle

def normalize_test(X_te, X_mu, X_sig):
    ''' Normalize test data features
    Args:
        X_te: Unnormalized training features
    Output:
        X_te: Normalized training features
    '''
    X_te = (X_te - X_mu)/X_sig
    return X_te 

with open("model/train_config", 'rb') as pickle_file:
    save_res = pickle.load(pickle_file)

loaded_model = load_model("model/model_deep_0.h5")
print("Loaded model from disk")
 
test_features = pickle.load(open("../feature_engg/feature/validation_features.p", "r") )
        
print len(test_features)

col_deleted=save_res["col_deleted"]
test_features = np.array(test_features)
test_features = np.delete(test_features, col_deleted, axis=1)

test_features = normalize_test(test_features, save_res["X_mu"], save_res["X_sig"])
print len(test_features)


# predict_proba outputs probability of each class
# [x, y] mean probability of class 0 is x and probability of class 1 is y
test_labels = loaded_model.predict_proba(test_features, verbose=1)

res = open("validate_label.csv", "w")

count = 0
with open("../../bytecup2016data/validate_nolabel.txt") as f:
    # # writing header
    res.write(f.readline())
    test_data = f.readline().strip()
    while test_data :        
        # probability of answering a question is probability in class 1
        prob = test_labels[count][1]
        res.write(test_data + "," + format(prob, '.8f') + "\n")
        count = count + 1
        test_data = f.readline().strip()
    
    
    
    
    
    

