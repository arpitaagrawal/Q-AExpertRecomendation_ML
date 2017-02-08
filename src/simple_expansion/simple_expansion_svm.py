'''
Uses flattened features in feature directory and run a SVM on it
'''

from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction import *
from sklearn.linear_model import *

import cPickle as pickle


features = pickle.load( open("./feature/simple_features.p", "rb") )
labels = pickle.load( open("./feature/labels.p", "rb") )

#     clf = svm.SVC()
clf = SGDClassifier(loss="hinge", penalty="l1")

clf.fit(features, labels)

# print clf.predict(features[0:10])
# print labels[0:10]

print joblib.dump(clf, './model/simple_expansion_svm.p')

print "done"    
    
