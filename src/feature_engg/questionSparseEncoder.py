from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2

import numpy as np
import cPickle as pickle
from __builtin__ import False

ques_new = []
updated_ques_data = []
ques_data_train = []
ques_data_test = []

## Hyper parameters for encoding question features:


def build_ques_data():
    from simple_expansion.simple_expansion_feature import get_ques_feature
     
    ques_id_dict = {}
    index =0       
    with open("../../bytecup2016data/question_info.txt") as f:
        question_data = f.readline().strip().split("\t")
        while question_data and len(question_data) == 7 :
            #ques_id = question_data[0]
            ques_id_dict[question_data[0]] = index
            ques_feature = get_ques_feature(question_data)
            ques_new.append(ques_feature)
            question_data = f.readline().strip().split("\t")
            index += 1
    print 'np.shape(ques_new)::', np.shape(ques_new)
    pickle.dump(ques_new, open("ques_complete_features.p", "wb"))   
    pickle.dump(ques_id_dict, open("question_ids_dict.p", "wb"))
    
    
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

#build_ques_data() 

ques_data = pickle.load( open("ques_complete_features.p", "rb") )
print np.shape(ques_data)

 
def train_ques_encoder_decoder(isNorm, loss_func, reg_p, enc_dim, file_suffix, batch):
    
    modify_data(isNorm) 
    num_of_features = np.shape(ques_data_train)[1]
    #call_ES = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')
    print 'number of features', num_of_features
    
    # "encoded" is the encoded representation of the input
    #encoded = Dense(enc_dim, activation='relu',W_regularizer=l2(reg_p))(input_features)
    # "decoded" is the lossy reconstruction of the input
#     decoded = Dense(num_of_features, activation='linear')(encoded)
    
    
    input_img = Input(shape=(num_of_features,))
    encoded = Dense(enc_dim*2, activation='relu',W_regularizer=l2(reg_p))(input_img)
    encoded = Dense(enc_dim, activation='relu',W_regularizer=l2(reg_p))(encoded)
    decoded = Dense(enc_dim*2, activation='relu')(encoded)
    decoded = Dense(num_of_features, activation='linear')(decoded)
     
    ##ENCODER MODEL for predictions
    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_img, output=decoded)
    encoder = Model(input=input_img, output=encoded)
    
    
    ##DECODER MODEL
    # create a placeholder for an encoded (32-dimensional) input
    #encoded_input = Input(shape=(enc_dim,))
    # retrieve the last layer of the autoencoder model
    #decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    #decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    ##DECODER MODEL END
    
    
    ksgd = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
    autoencoder.compile(optimizer=ksgd, loss=loss_func)
     
    hist = autoencoder.fit(ques_data_train, ques_data_train,
                    nb_epoch=100,
                    batch_size=batch,
                    shuffle=True,
                    validation_data=(ques_data_test, ques_data_test))
    
    print "history", hist.history
    
    with open("ques_loss_history.txt", "a+") as loss_history:
        d = str(hist.history)
        loss_history.write(d)
    
    model_file_name = "ques_encoder_model_deep_"+str(file_suffix)+".h5"
    autoencoder.save(model_file_name) 
    
    compressed_features = encoder.predict(updated_ques_data)
    compressed_featurs_name = "encodedFeatures/compressed_ques_features_"+str(file_suffix)+".p"
    pickle.dump(compressed_features, open(compressed_featurs_name, "wb+"))
     
#train_ques_encoder_decoder()

def modify_data(isNorm):
    
    global updated_ques_data
    global ques_data_train, ques_data_test
    if isNorm:
        [updated_ques_data, data_mu, data_sig] = normalize(ques_data)
#         with open("parameters.txt", "a") as myfile:
#             d = str(isNorm), ",sigma: ", str(data_sig), ", mean::", data_mu
#             myfile.write( str(d))
    else:
        updated_ques_data = ques_data
    
    #num_rows = np.shape(updated_ques_data)[0]
    #updated_ques_data = np.column_stack(( np.ones(num_rows), updated_ques_data))

    ques_data_train = updated_ques_data[0:8000]
    ques_data_test = updated_ques_data[8000:8095]
    
    
def gridSearch():
    normalise_value = [True]
    loss_func_arr = ['mse']
    reg_param = [0.001,0.01,0.1]
    encoding_dim = [500]
    batch_size = [700]
    file_suffix = 0
    
    with open("ques_parameters.txt", "a+") as myfile:
        
        for isNorm in normalise_value:
            
            for loss_func in loss_func_arr:
                for reg_p in reg_param:
                    for enc_dim in encoding_dim:
                        for batch in batch_size:
                            data =data ="\n\n\n" + str(isNorm) + "  ,Loss: "+ str(loss_func)+ "  ,Reg_param "+ str(reg_p)+ "  ,Encoding_dim "+ str(enc_dim)+ "  ,batch size: "+ str(batch)+", file_suffix::"+ str(file_suffix)
                            print data
                            myfile.write(str(data))
                            train_ques_encoder_decoder(isNorm, loss_func, reg_p, enc_dim, file_suffix, batch)
                            file_suffix += 1
                break
                            
gridSearch()
