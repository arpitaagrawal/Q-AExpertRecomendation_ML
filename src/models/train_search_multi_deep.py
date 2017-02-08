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
from sklearn.metrics import f1_score

import theano
from models.down_sampling import balanced_subsample
theano.config.openmp = True
OMP_NUM_THREADS=16 

users_va_te_dict = dict([ (v,idx) for (idx,v) in enumerate(pickle.load(open("../../bytecup2016data/users_va_te.p"))) ])

print "users_va_te_dict created ", len(users_va_te_dict)

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
    labels_new_arr = []
    for idx,label in enumerate(labels):
        label_new = [0] * len(users_va_te_dict) * 2
        
        if label[1] == '0' :
            label_new[ users_va_te_dict[label[0]] ] = 1
        else :
            label_new[ users_va_te_dict[label[0]] + 1 ] = 1
        
        labels_new_arr.append(label_new)
        
#         if (idx+1) % 1000 == 0:
#             break
        
        
    print "labels_new_arr created" , len(labels_new_arr)
    return labels_new_arr

def original_label(label):
    return [ l.index(1) for l in label]

def get_transform_label():
    '''
    Returns list of labels as list of [0/1 , 1/0] 
    if label = 1 [0, 1]
    if label = 0 [1, 0]
    '''
    count = 0
    users_order = []
    ##features to be deletd
    del_rows = []
    with open("../../bytecup2016data/invited_info_train_PROC.txt","r") as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) >= 2 :
            user_id = training_data[1]
            label = training_data[2]
            
            if user_id in users_va_te_dict:
                users_order.append((user_id,label) )
            else:
                del_rows.append(count)
                count += 1
            training_data = f.readline().strip().split("\t")
        f.close()
    
    print "users_order created ", len(users_order), len(del_rows)
    return transform_label(users_order), del_rows
 
features = pickle.load( open("../feature_engg/feature/all_features.p", "rb") )
labels, del_rows = get_transform_label()

# features = np.random.normal(size=(26796,3))
# labels, del_rows = get_transform_label()

print len(features),len(features[0])
print len(labels),len(labels[0])

features = np.array(features)

features = np.delete(features, del_rows, axis=0)

col_deleted = np.nonzero((features==0).sum(axis=0) > (len(features)-1000))
# col_deleted = col_deleted[0].tolist() + range(6,22) + range(28,44)
print col_deleted
features = np.delete(features, col_deleted, axis=1)

print len(features),len(features[0])
print len(labels),len(labels[0])

features, X_mu, X_sig = normalize(features)

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

features,labels = [] , []

features_tr, features_te,labels_tr, labels_te = train_test_split(features,labels, train_size = 0.85)
print "Using separate test data", len(features_tr), len(features_te)

def run_NN(arch, reg_coeff, sgd_decay, subsample_size=0, save=False):
#     features_tr, labels_tr = balanced_subsample(features_tr, original_label(labels_tr), subsample_size = subsample_size)
#     labels_tr = transform_label(labels_tr)
#     print "Training data balanced-", features_tr.shape, len(labels_tr)
        
    call_ES = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='auto')
    
    # Generate Model
    model = genmodel(num_units=arch, reg_coeff=reg_coeff )
    # Compile Model
    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=momentum, 
        nesterov=sgd_Nesterov)
    
    # sgd = RMSprop(lr=sgd_lr, rho=0.9, epsilon=1e-08, decay=sgd_decay)
    
    model.compile(loss='MSE', optimizer=sgd, 
        metrics=['accuracy'])
    # Train Model
    if eStop:
        model.fit(features_tr, labels_tr, nb_epoch=nb_epoch, batch_size=batch_size, 
        verbose=verbose, callbacks=[call_ES], validation_split=0.1, 
        validation_data=None, shuffle=True)
    else:
        model.fit(features_tr, labels_tr, nb_epoch=nb_epoch, batch_size=batch_size, 
            verbose=verbose)
    
    labels_pred = model.predict_classes(features_te)
    print len(labels_te[0]), labels_pred[0]
    y_true, y_pred = original_label(labels_te), labels_pred
    
    print y_true[0], y_pred[0]
    print "arch, reg_coeff, sgd_decay, subsample_size", arch, reg_coeff, sgd_decay, subsample_size

    macro_rep = f1_score(y_true, y_pred, average = 'macro')
    print "macro", macro_rep
    
    weighted_report = f1_score(y_true, y_pred, average = 'weighted')
    print "weighted", weighted_report
    with open("results_search_multi_deep.txt", "a") as f:
        f.write("macro_rep- "+str(macro_rep))
        f.write("\n")
        f.write("weighted_report- "+str(weighted_report))
        f.write("\n")
        f.write(" ".join([str(s) for s in ["arch, reg_coeff, sgd_decay, subsample_size", arch, reg_coeff, sgd_decay, subsample_size]]))
        f.write("\n")
        
    if save:
        # Save model
        model.save("model/model_deep.h5")
        print("Saved model to disk")
    

 
arch_range = [[len(features_tr[0]),1024,len(labels_tr[0])], [len(features_tr[0]),1024,512,len(labels_tr[0])], [len(features_tr[0]),1024,1024,len(labels_tr[0])],[len(features_tr[0]),1024,512,256,len(labels_tr[0])]]
reg_coeffs_range = [1e-6, 5e-6, 1e-5, 5e-5, 5e-4 ]
sgd_decays_range = [1e-6, 1e-5, 5e-5, 1e-4, 5e-4 ]
class_weight_0_range = [1]
# subsample_size_range = [2,2.5,3]

#GRID SEARCH ON BEST PARAM
for arch in arch_range:
    for reg_coeff in reg_coeffs_range:
        for sgd_decay in sgd_decays_range:
#             for subsample_size in subsample_size_range:
            run_NN(arch, reg_coeff, sgd_decay)

# arch = [len(features[0]),1024,512,2]
# reg_coeff = 1e-05
# sgd_decay = 1e-05
# class_weight_0 = 0.5
 

