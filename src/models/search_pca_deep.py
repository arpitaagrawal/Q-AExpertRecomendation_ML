'''
Uses flattened features in feature directory and run a SVM on it
'''

from keras.layers import Dense
from keras.models import Sequential
import keras.regularizers as Reg
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
import cPickle as pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import theano
from models.down_sampling import balanced_subsample
theano.config.openmp = True
OMP_NUM_THREADS=24 

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



def genmodel(num_units, actfn='relu', reg_coeff=0.0, last_act='softmax'):
    ''' Generate a neural network model of approporiate architecture
    Args:
        num_units: architecture of network in the format [n1, n2, ... , nL]
        actfn: activation function for hidden layers ('relu'/'sigmoid'/'linear'/'softmax')
        reg_coeff: L2-regularization coefficient
        last_act: activation function for final layer ('relu'/'sigmoid'/'linear'/'softmax')
    Output:
        model: Keras sequential model with appropriate fully-connected architecture
    '''

    model = Sequential()
    for i in range(1, len(num_units)):
        if i == 1 and i < len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=actfn, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == 1 and i == len(num_units) - 1:
            model.add(Dense(input_dim=num_units[0], output_dim=num_units[i], activation=last_act, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i < len(num_units) - 1:
            model.add(Dense(output_dim=num_units[i], activation=actfn, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
        elif i == len(num_units) - 1:
            model.add(Dense(output_dim=num_units[i], activation=last_act, 
                W_regularizer=Reg.l2(l=reg_coeff), init='glorot_normal'))
    return model

def transform_label(labels):
    labels_new = []
    for label in labels:
        label_new = [0.0,0.0]
        label_new[int(label)]=1.0
        labels_new.append(label_new)
    
    return labels_new

def original_label(label):
    return [ 0*l[0] + 1*l[1] for l in label]

def get_transform_label():
    '''
    Returns list of labels as list of [0/1 , 1/0] 
    if label = 1 [0, 1]
    if label = 0 [1, 0]
    '''
    return transform_label(pickle.load( open("../feature_engg/feature/labels.p", "rb") ) )
    

features_orig = pickle.load( open("../feature_engg/feature/all_features.p", "rb") )
labels_orig = get_transform_label()

# features_orig = np.random.normal(size=(2294,354))
# labels_orig = [[1.0,0.0]]*2000 + [[0.0,1.0]]*(2294-2000)

print len(features_orig),len(features_orig[0])
print len(labels_orig),len(labels_orig[0])

features_orig = np.array(features_orig)

col_deleted = np.nonzero((features_orig==0).sum(axis=0) > (len(features_orig)-1000))
# col_deleted = col_deleted[0].tolist() + range(6,22) + range(28,44)
print col_deleted
features_orig = np.delete(features_orig, col_deleted, axis=1)

print len(features_orig),len(features_orig[0])
print len(labels_orig),len(labels_orig[0])

features_orig, X_mu, X_sig = normalize(features_orig)

save_res = {"col_deleted":col_deleted,"X_mu":X_mu,"X_sig":X_sig}
with open("model/train_config", 'wb') as pickle_file:
    pickle.dump(save_res, pickle_file, protocol=2)
print "Dumped config"

momentum = 0.99
eStop = True
sgd_Nesterov = True
sgd_lr = 1e-5
batch_size=5000
nb_epoch=100
verbose=True

model_num=20 

def run_NN(features,labels, arch, reg_coeff, sgd_decay, class_weight_0,subsample_size=2.0, save=False, test=True):
    '''
    Runs NN with give params obtained from grid search. 
    If save is enabled - Runs and saves model on full training set
    If save is disable - Takes out a test data and runs on reamaing training set. Prints a classification report.
     
    '''
    print "<run_NN>"
    global model_num
    if test:
        features_tr, features_te,labels_tr, labels_te = train_test_split(features,labels, train_size = 0.8, random_state=32)
        print "Using separate test data", len(features_tr), len(features_te)
        
        features_tr, labels_tr = balanced_subsample(features_tr, original_label(labels_tr), subsample_size=subsample_size)
        labels_tr = transform_label(labels_tr)
        print "Training data balanced-", features_tr.shape, len(labels_tr)
    else:
        features_tr, labels_tr, features_te, labels_te =  balanced_subsample(features, original_label(labels), subsample_size=subsample_size), features[0:1000], labels[0:1000] 
        labels_tr = transform_label(labels_tr)
        print "Using a sample training data"
    
    call_ES = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='max')
    
    # Generate Model
    model = genmodel(num_units=arch, reg_coeff=reg_coeff )
    # Compile Model
    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=momentum, 
        nesterov=sgd_Nesterov)
    
    # sgd = RMSprop(lr=sgd_lr, rho=0.9, epsilon=1e-08, decay=sgd_decay)
    
    model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
    # Train Model
    if eStop:
        model.fit(features_tr, labels_tr, nb_epoch=nb_epoch, batch_size=batch_size, 
        verbose=verbose, callbacks=[call_ES], validation_split=0.1, 
        validation_data=None, shuffle=True, class_weight={0:class_weight_0 , 1:1})
    else:
        model.fit(features_tr, labels_tr, nb_epoch=nb_epoch, batch_size=batch_size, 
            verbose=verbose, class_weight={0: class_weight_0 , 1:1})
    
    if not save:
        labels_pred = model.predict_classes(features_te)
        print labels_te[0], labels_pred[0]
        y_true, y_pred = [ 0*l[0] + 1*l[1] for l in labels_te], labels_pred
        
        print y_true[0], y_pred[0]
        print "arch, reg_coeff, sgd_decay, class_weight_0", arch, reg_coeff, sgd_decay, class_weight_0
        
        report = classification_report(y_true, y_pred)
        print report
        with open("results_pca_search.txt", "a") as f:
            f.write(report)
            f.write("\n")
            f.write(" ".join([str(s) for s in ["arch, reg_coeff, sgd_decay, class_weight_0", arch, reg_coeff, sgd_decay, class_weight_0]]))
            f.write("\n")
        
    if save:
        # Save model
        print arch, reg_coeff, sgd_decay, class_weight_0,subsample_size, save
        model.save("model/model_deep_{0}.h5".format(model_num))
        print "Saved model to disk", "model/model_deep_{0}.h5".format(model_num)
        model_num+=1
    
    print "</run_NN>"



from sklearn import decomposition

orig_length = len(features_orig[0])

arch_range = [[len(features_orig[0]),512,2], [len(features_orig[0]),1024,2], [len(features_orig[0]),1024,512,2], [len(features_orig[0]),1024,1024,2],  [len(features_orig[0]), 512, 1024, 512,2]]
reg_coeffs_range = [5e-6, 1e-5, 5e-5, 5e-4, 1e-2 ]
sgd_decays_range = [1e-6, 1e-5, 5e-5, 1e-4,  1e-2 ]
comp_range = range(orig_length/4,orig_length/2,10)

for arch in arch_range:
    for reg_coeff in reg_coeffs_range:
        for sgd_decay in sgd_decays_range:
            for component in comp_range :
                print component
                pca = decomposition.PCA(n_components=component)
                pca.fit(features_orig)
                X = pca.transform(features_orig)
                
                # Adding bias
                X = np.column_stack((np.ones(len(X)),X))
                # Making length consistent
                arch[0] = len(X[0])
                run_NN(X, labels_orig,arch, reg_coeff, sgd_decay, 1, 2.5, save=False, test=True)
                run_NN(X, labels_orig,arch, reg_coeff, sgd_decay, 1, 2, save=False, test=True)
                



