import cPickle as pickle
from feature_engg.create_user_question_matrix import user_to_idx,ques_to_idx
from  simple_expansion.simple_expansion_feature import users, questions

folder = "../clusters/dist_data/"

if __name__ == '__main__':
    users_all_idx = dict( [(val,idx) for idx,val in enumerate(users.keys())])
    questions_all_idx = dict( [(val,idx) for idx,val in enumerate(questions.keys())])
    print "Saving users_all_idx, questions"
    pickle.dump(users_all_idx, open(folder + "users_all_idx.p","w"), protocol=2)
    pickle.dump(questions_all_idx, open(folder + "questions_all_idx.p","w"), protocol=2)
    
    ## Below should be used saved on same data as per user_to_ques
    ## This is done for hamming distance on get_distance_metric_col.py
    print "Saving user_idx and question_idx"
    pickle.dump(ques_to_idx, open(folder+"question_train_idx.p","w"), protocol=2)
    pickle.dump(user_to_idx, open(folder + "user_train_idx.p","w"), protocol=2)


users_all_idx = pickle.load(open(folder + "users_all_idx.p","r"))
questions_all_idx = pickle.load(open(folder + "questions_all_idx.p","r"))

question_train_idx = pickle.load(open(folder+"question_train_idx.p","r"))
user_train_idx = pickle.load( open(folder + "user_train_idx.p","r"))
    
    
print "users_all_idx",len(users_all_idx)
print "user_train_idx",len(user_train_idx)
print "questions_all_idx",len(questions_all_idx)
print "question_train_idx", len(question_train_idx)
