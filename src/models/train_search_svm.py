'''
Uses flattened features in feature directory and run a SVM on it
'''

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import cPickle as pickle
from sklearn.metrics import f1_score, make_scorer
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

def score_f1_class1(ground_truth, predictions):
    '''
    Returns f1 score for class 1
    '''
    return f1_score(ground_truth, predictions, average='binary', pos_label=1)

features = pickle.load( open("../feature_engg/feature/all_features.p", "rb") )
labels = [int(l) for l in pickle.load( open("../feature_engg/feature/labels.p", "rb") )]


print len(features),len(features[0])
print len(labels),labels[0]

features = np.array(features)

col_deleted = np.nonzero((features==0).sum(axis=0) > (len(features)-1000))
print col_deleted
features = np.delete(features, col_deleted, axis=1)

print len(features),len(features[0])
print len(labels),labels[0]

features, X_mu, X_sig = normalize(features)

gamma_ramge = [ 4**i for i in range(-3,0) ]
C_range = [ 4**i for i in range(-1,6) ]
class_weight_range = [ {0: class_weight_0 , 1:1} for class_weight_0 in [0.95]]

score = make_scorer(score_f1_class1, greater_is_better=True)

# Number of folds in Cross validation
CV_FOLDS = 3
# Number of parallel jobs
parallel = 14

X_tr, X_te,y_tr, y_te = train_test_split(features,labels, train_size = 0.85)

svr = SVC()

parameters = [{ 'kernel':['rbf'], 'C':C_range, 'gamma':gamma_ramge,'class_weight':class_weight_range
               ,'cache_size':[1000], 'tol':[1e-2], 'max_iter':[1000]}]

clf = GridSearchCV(svr, parameters, cv=CV_FOLDS, n_jobs = parallel, verbose=1000, iid=False, scoring=score)
print "data splitted for testing ", len(y_tr), len(y_te) 
    
X_tr, y_tr = balanced_subsample(X_tr, y_tr, subsample_size=2.5)
print "Training data balanced-", X_tr.shape, len(y_tr)

clf.fit(X_tr,y_tr)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
mean_fit_time = clf.cv_results_['mean_fit_time']

for mean, std, params,time in zip(means, stds, clf.cv_results_['params'],mean_fit_time):
    print("%0.3f (+/-%0.03f) for %r time-%0.3f"
          % (mean, std * 2, params,time))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_te, clf.predict(X_te)
print(classification_report(y_true, y_pred))
print()

print "done"    
    
