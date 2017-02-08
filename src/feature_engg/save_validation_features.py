from feature_engg import create_features as eng_feat 
import cPickle as pickle

validation_features = []
with open("../../bytecup2016data/validate_nolabel.txt") as f:
    # skip header for prediction
    f.readline()
    test_data = f.readline().strip().split(",")

    while test_data and len(test_data) == 2 :
        question_id = test_data[0]
        user_id = test_data[1]
        
        feature = eng_feat.get_full_feature(question_id, user_id)
        validation_features.append(feature)
        
        test_data = f.readline().strip().split(",")
        

print len(validation_features), len(validation_features[0])

pickle.dump(validation_features, open("./feature/validation_features.p", "wb"), protocol=2 )