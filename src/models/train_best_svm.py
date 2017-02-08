'''
Uses flattened features in feature directory and run a SVM on it
'''

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import cPickle as pickle
from models.down_sampling import balanced_subsample

def normalize(X_tr):
    ''' Normalize training and test data features
    Args:
        X_tr: Unnormalized training features
    Output:
        X_tr: Normalized training features
    '''
    X_mu = np.mean(X_tr, axis=0)
    X_tr = X_tr - X_mu
    X_sig = np.std(X_tr, axis=0)
    X_tr = X_tr/X_sig
    return X_tr, X_mu, X_sig


features = pickle.load( open("../feature_engg/encodedFeatures/encoded_completed_features.p", "rb") )
labels = [int(l) for l in pickle.load( open("../feature_engg/encodedFeatures/labels.p", "rb") )]

# features = np.random.normal(size=(2294,354))
# labels = [0]*2000 + [1]*(2294-2000)

print len(features),len(features[0])
print len(labels),labels[0]

features = np.array(features)

col_deleted = np.nonzero((features==0).sum(axis=0) > (len(features)-1000))
print col_deleted
features = np.delete(features, col_deleted, axis=1)

print len(features),len(features[0])
print len(labels),labels[0]

features, X_mu, X_sig = normalize(features)
print "data normalised"

save_res = {"col_deleted":col_deleted,"X_mu":X_mu,"X_sig":X_sig}
with open("model/train_config_svm.p", 'wb') as pickle_file:
    pickle.dump(save_res, pickle_file, protocol=2)
print "Dumped config"


save = True

if not save:
    '''
    Split into test and train and evaluate test   
    '''
    X_tr, X_te,y_tr, y_te = train_test_split(features,labels, train_size = 0.85)
    print "data splitted for testing ", len(y_tr), len(y_te) 
    
    X_tr, y_tr = balanced_subsample(X_tr, y_tr, subsample_size=2.0)
    print "Training data balanced-", X_tr.shape, len(y_tr)

    clf = SVC(kernel='rbf',C=16,gamma=0.015625,class_weight={0: 1 , 1:1}, cache_size=1000, tol=1e-1, max_iter=100, verbose=True)
    
    clf.fit(X_tr,y_tr)
    
    print("Detailed classification report:")
    print("Model restricted to 100 iteration")
    print ""
    y_true, y_pred = y_te, clf.predict(X_te)
    print(classification_report(y_true, y_pred))
    
    print "Now running with restricting to 2000 iteration"    
        
    clf = SVC(kernel='rbf',C=16,gamma=0.015625,class_weight={0: 1 , 1:1}, cache_size=1000, tol=1e-1, max_iter=2000, verbose=True)
    
    clf.fit(X_tr,y_tr)
    
    
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_te, clf.predict(X_te)
    print(classification_report(y_true, y_pred))
    print()
    
    print "done"   
else:
    X_tr, y_tr = balanced_subsample(features, labels, subsample_size=2.0)

    clf = SVC(kernel='rbf',C=16,gamma=0.015625,class_weight={0: 1 , 1:1}, cache_size=1000, tol=1e-1, max_iter=200, verbose=True, probability=True)
    
    clf.fit(X_tr,y_tr)
    
#     pickle.dump(clf, open("./model/model_svm_dsample.p","w"), protocol=2)
    
# StandardScaler or MinMaxScaler.