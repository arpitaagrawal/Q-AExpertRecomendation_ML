import cPickle

def convert_to_csv():
    '''
    Convert ques_id, user_id to user int id, ques int id
    stores a mapping of id to int id
    '''
    mahout_file = open("../bytecup2016data/invited_info_train_mahout.csv","w")
    mahout_test_file = open("../bytecup2016data/test_nolabel_mahout.csv","w")
    
    # variable to construct user and ques features
    max_user = 0
    user_to_idx = {}
    max_ques = 0
    ques_to_idx = {}

    with open("../bytecup2016data/invited_info_train_PROC.txt","r") as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            question_id = training_data[0]
            user_id = training_data[1]
            label = training_data[2]
            
            if user_id not in user_to_idx: 
                user_to_idx[user_id] = max_user
                max_user+=1
                 
            if question_id not in ques_to_idx: 
                ques_to_idx[question_id] = max_ques
                max_ques+=1
            
            mahout_file.write( "{0},{1},{2}\n".format(user_to_idx[user_id], ques_to_idx[question_id], label) )
#             if label == "1":  
#                 mahout_file.write( "{0},{1}\n".format(user_to_idx[user_id], ques_to_idx[question_id], label) )
                            
            training_data = f.readline().strip().split("\t")
        f.close()
        mahout_file.close()
    
    #invited_info_train_PROC_test
    with open("../bytecup2016data/validate_nolabel.txt","r") as f:
        training_data = f.readline().strip().split(",")
        training_data = f.readline().strip().split(",")
        while training_data and len(training_data) >= 2 :
            question_id = training_data[0]
            user_id = training_data[1]
            
            if user_id not in user_to_idx: 
                user_to_idx[user_id] = max_user
                max_user+=1
                 
            if question_id not in ques_to_idx: 
                ques_to_idx[question_id] = max_ques
                max_ques+=1
            
            mahout_test_file.write( "{0},{1}\n".format(user_to_idx[user_id], ques_to_idx[question_id], label) )
                            
            training_data = f.readline().strip().split(",")
        f.close()
        mahout_test_file.close()
    
    print max_user, max_ques
    cPickle.dump(user_to_idx, open("../bytecup2016data/user_to_idx.p","wb"), protocol=2)
    cPickle.dump(ques_to_idx, open("../bytecup2016data/ques_to_idx.p","wb"), protocol=2)
    
        
if __name__ == '__main__':
    convert_to_csv()
    print "Finished writing mahout file"
    
    