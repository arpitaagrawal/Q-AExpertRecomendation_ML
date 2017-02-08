'''
Uses flattened features in feature directory and run a SVM on it
'''
 
from keras.models import load_model
import numpy as np
import cPickle as pickle
from keras.preprocessing import sequence

loaded_model_char = load_model("model/rnn_char.h5")
print("Loaded rnn_char model from disk")
loaded_model_tag = load_model("model/rnn_tag.h5")
print("Loaded rnn_tag model from disk")
loaded_model_words = load_model("model/rnn_words.h5")
print("Loaded rnn_words model from disk")
 
test_features_char = pickle.load(open("../feature_engg/feature/char_features.p", "r") )
test_features_tag = pickle.load(open("../feature_engg/feature/tag_features.p", "r") )
test_features_word = pickle.load(open("../feature_engg/feature/word_features.p", "r") )

print("Loaded all features from disk")

len_features_char = 130
len_features_tag = 10
len_features_word = 60

loaded_models = [loaded_model_char, loaded_model_tag, loaded_model_words]
test_features = [test_features_char, test_features_tag, test_features_word]
len_features = [len_features_char, len_features_tag,len_features_word]

res_file_prefix = "rnn_tr_"
res_files = [res_file_prefix + "char.csv", res_file_prefix + "tag.csv", res_file_prefix + "word.csv"]


for loaded_model,test_feature,res_file,len_feature in zip(loaded_models,test_features,res_files,len_features):
    # predict_proba outputs probability of each class
    # [x, y] mean probability of class 0 is x and probability of class 1 is y
    
    test_feature = sequence.pad_sequences(test_feature, maxlen=len_feature)
    test_labels = loaded_model.predict_proba(test_feature, verbose=1)
    
    print test_labels[0]
    res = open(res_file, "w")
    
    count = 0
    with open("../../bytecup2016data/invited_info_train_test.txt") as f:
        # # writing header
        res.write(f.readline())
        test_data = f.readline().strip()
        while test_data :        
            # probability of answering a question is probability in class 1
            prob = test_labels[count][0]
            
            td = test_data.split(",")
            res.write(td[0] + "," + td[1] + "," + format(prob, '.8f') + "\n")
            count = count + 1
            test_data = f.readline().strip()
        
        
    
    
    
    

