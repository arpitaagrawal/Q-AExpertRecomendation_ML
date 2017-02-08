'''
Uses flattened features in feature directory and run a SVM on it
'''

from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction import *
from sklearn.linear_model import *

from simple_expansion_feature import questions, users, get_full_feature 

clf = joblib.load('./model/simple_expansion_svm.p')

test_features = []
with open("../../bytecup2016data/validate_nolabel.txt") as f:
    # skip header for prediction
    f.readline()
    test_data = f.readline().strip().split(",")

    while test_data and len(test_data) == 2 :
        question = questions[test_data[0]]
        user = users[test_data[1]]
        
        feature = get_full_feature(question, user)
        test_features.append(feature)
        
        test_data = f.readline().strip().split(",")
        
        
print len(test_features)

test_labels = clf.predict(test_features)

res = open("validate_label.csv", "w")

count = 0
with open("../../bytecup2016data/validate_nolabel.txt") as f:
    # # writing header
    res.write(f.readline())
    test_data = f.readline().strip()
    while test_data :
        res.write(test_data + "," + str(test_labels[count]) + "\n")
        count = count + 1
        test_data = f.readline().strip()
    
    
    
    
    
    

