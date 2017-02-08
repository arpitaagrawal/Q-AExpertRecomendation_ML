from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.optimizers import SGD

import numpy as np
import cPickle as pickle

users_new = []

updated_user_data = []
user_data_train = []
user_data_test = []
myfile = ''

def build_user_data():
    users_id_dict = {}
    from simple_expansion.simple_expansion_feature import users, get_user_feature
    index = 0
    with open("../../bytecup2016data/user_info.txt") as f:
        user_data = f.readline().strip().split("\t")
        while user_data and len(user_data) == 4 :
            user_id = user_data[0]
            users_id_dict[user_id] = index
            user_feature = get_user_feature(user_data)
            users_new.append(user_feature)
            user_data = f.readline().strip().split("\t")
            index += 1
     
    print 'np.shape(users_new)::', np.shape(users_new)
    pickle.dump(users_new, open("user_complete_features.p", "wb"))   
    pickle.dump(users_id_dict, open("users_ids_dict.p", "wb"))
#     print users_new[0]
#build_user_data() 


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

user_data = pickle.load( open("user_complete_features.p", "rb") )
print np.shape(user_data)



def train_user_encoder_decoder(isNorm, loss_func, reg_p, lr, enc_dim, file_suffix, batch):
    
    modify_data(isNorm) 
    num_of_features = np.shape(user_data_train)[1]
    
    print 'number of features', num_of_features
    input_features = Input(shape=(num_of_features,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(enc_dim, activation='relu',W_regularizer=regularizers.l2(reg_p), activity_regularizer=regularizers.activity_l2(0.01))(input_features)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(num_of_features, activation='linear')(encoded)
    
    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_features, output=decoded)
    encoder = Model(input=input_features, output=encoded)
    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(enc_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
    
    ksgd = SGD(lr=lr, decay=1e-6, momentum=0.99, nesterov=True)
    autoencoder.compile(optimizer=ksgd, loss=loss_func)
    #autoencoder.compile(optimizer='adadelta', loss=loss_func)
    
    hist = autoencoder.fit(user_data_train, user_data_train,
                    nb_epoch=100,
                    batch_size=batch,
                    shuffle=True,
                    validation_data=(user_data_test, user_data_test))
    
    with open("user_loss_history.txt", "a+") as loss_history:
        d = str(hist.history)
        print 'd', d
        loss_history.write(d)
    
    model_file_name = "user_encoder_model_deep_"+str(file_suffix)+".h5"
    autoencoder.save(model_file_name)
    
    #print "user_data_test first training example:", user_data_test[0]
    compressed_features = encoder.predict(updated_user_data)
    compressed_featurs_name = "encodedFeatures/compressed_user_features_"+str(file_suffix)+".p"
    pickle.dump(compressed_features, open(compressed_featurs_name, "wb+"))
    
    #print compressed_features[0]
    #decoded_imgs = decoder.predict(compressed_features)
    #print '\n', decoded_imgs[0, 0:30]

def modify_data(isNorm):
    
    global updated_user_data
    global user_data_train, user_data_test
    if isNorm:
        [updated_user_data, data_mu, data_sig] = normalize(user_data)
#         with open("parameters.txt", "a") as myfile:
#             d = str(isNorm), ",sigma: ", str(data_sig), ", mean::", data_mu
#             myfile.write( str(d))
    else:
        updated_user_data = user_data

    num_rows = np.shape(updated_user_data)[0]
    
    updated_user_data = np.column_stack(( np.ones(num_rows), updated_user_data))

    user_data_train = updated_user_data[0:27000]
    user_data_test = updated_user_data[27000:28763]
    
    

def gridSearch():
    normalise_value = [True, False]
    loss_func_arr = ['binary_crossentropy', 'mse','kld']
    reg_param = [0.00005, 0.00001, 0.0001, 0.001, 0.3, 0.1, 1,2]
    learnin_param = [0.005,0.001,0.0005, 0.05, 0.01]
    encoding_dim = [100,300,500,1000]
    batch_size = [1000]
    file_suffix = 0
    global myfile
    
    with open("user_parameters.txt", "a+") as myfile:
        
        for isNorm in normalise_value:
            
            for loss_func in loss_func_arr:
                for reg_p in reg_param:
                    for lr in learnin_param:
                        for enc_dim in encoding_dim:
                            for batch in batch_size:
                                data ="\n\n\n" + str(isNorm) + "  ,Loss: "+ str(loss_func)+ "  ,Reg_param "+ str(reg_p)+ "  ,Encoding_dim "+ str(enc_dim)+ "  ,batch size: "+ str(batch)+" , file_suffix::"+ str(file_suffix)
                                print data
                                myfile.write(str(data))
                                train_user_encoder_decoder(isNorm, loss_func, reg_p, lr, enc_dim, file_suffix, batch)
                                file_suffix += 1
                            
gridSearch()

