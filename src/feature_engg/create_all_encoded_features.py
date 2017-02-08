import simple_expansion.simple_expansion_feature as simp
import cPickle as pickle
import numpy as np
training_data = []
labels = []
  
encoded_ques_features = pickle.load( open("encodedFeatures/compressed_ques_features.p", "rb"))
encoded_user_features = pickle.load( open("encodedFeatures/compressed_user_features.p", "rb"))  
ques_id_dict = pickle.load( open("question_ids_dict.p", "rb"))
print len(ques_id_dict), 'len of dict'
user_id_dict = pickle.load( open("users_ids_dict.p", "rb"))

def create_encoded_feature(user_id, ques_id):
    
    full_encoded_feature_arr = []
    ques_index = ques_id_dict[ques_id]        
    user_index = user_id_dict[user_id]
    ques_feature = encoded_ques_features[ques_index] 
    user_feature = encoded_user_features[user_index]
    full_encoded_feature_arr.extend(ques_feature)
    full_encoded_feature_arr.extend(user_feature)
    
    return full_encoded_feature_arr

def transform_validation_feature():
    test_features = []
    with open("../../bytecup2016data/validate_nolabel.txt") as f:
    # skip header for prediction
        f.readline()
        test_data = f.readline().strip().split(",")
    
        while test_data and len(test_data) == 2 :
            question_id = test_data[0]
            user_id = test_data[1]
            
            feature = create_encoded_feature(question_id, user_id)
            test_features.append(feature)
            
            test_data = f.readline().strip().split(",") 
    print len(test_features)
    pickle.dump(test_features, open("encodedFeatures/encoded_no_label_data.p", "wb")) 
    

def transform_test_data():
    count = 0
    
    with open(simp.INVITED_INFO_TRAIN) as f:
        training_data_line = f.readline().strip().split("\t")
        
        while training_data_line and len(training_data_line) == 3 :
            count = count +1
            
            label = training_data_line[2]
            question_id = training_data_line[0]
            user_id = training_data_line[1]
            full_encoded_feature_list = create_encoded_feature(user_id, question_id)
            training_data.append(full_encoded_feature_list)
            labels.append(label)
            training_data = f.readline().strip().split("\t")
            count += 1
            if(count % 1000) == 0:
                print count, "processed"
    
    pickle.dump(training_data, open("encodedFeatures/encoded_completed_features.p", "wb"))
    pickle.dump(labels, open("encodedFeatures/labels.p", "wb"))

if __name__ == '__main__':
    #transform_test_data()
    #transform_validation_feature()
    encoded_no_label_data = pickle.load( open("encodedFeatures/encoded_no_label_data.p", "rb"))
    
    

    
    