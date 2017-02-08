'''
This file will compute a matrix USER X MATRIX based on questions a user answered
'''
from simple_expansion import simple_expansion_feature as simp
import numpy as np

# variable to construct user and ques features
max_user = 0
user_to_idx = {}
max_ques = 0
ques_to_idx = {}

# matrix for users to questions
user_to_ques = []

# tot_ques and tot_users added after calculation       
tot_ques = -1
tot_users = -1

with open(simp.INVITED_INFO_TRAIN) as f:
    training_data = f.readline().strip().split("\t")
    while training_data and len(training_data) == 3 :
        label = training_data[2]
        question_id = training_data[0]
        user_id = training_data[1]
        
        #Add user to dict if empty
        if user_id not in user_to_idx: 
            user_to_idx[user_id] = max_user
            max_user+=1
             
        if question_id not in ques_to_idx: 
            ques_to_idx[question_id] = max_ques
            max_ques+=1

        training_data = f.readline().strip().split("\t")
    
    f.close()

tot_ques = len(ques_to_idx)
tot_users = len(user_to_idx)

user_to_ques = np.zeros((tot_users,tot_ques))
print "user_to_ques Matrix size ", user_to_ques.shape

with open(simp.INVITED_INFO_TRAIN) as f:
    training_data = f.readline().strip().split("\t")
    while training_data and len(training_data) == 3 :
        label = int(training_data[2])
        question_idx = ques_to_idx[training_data[0]]
        user_idx = user_to_idx[training_data[1]]
    
        user_to_ques[user_idx][question_idx] = label 
                    
        training_data = f.readline().strip().split("\t")
    
    f.close()


# np.save("feature/user_to_ques", user_to_ques)
# 
# print "saved in feature/user_to_ques.npy"

def get_user_feature(user_id):
    '''
    Returns a array of 0/1 for a user depending on question answered / not
    '''    
    if user_id not in user_to_idx:
        return np.zeros((1,tot_ques))[0]
    user_idx = user_to_idx[user_id]
    
    return user_to_ques[user_idx]


def get_ques_feature(question_id):
    '''
    Returns a array of 0/1 for a question depending on user answered / not
    '''
    if question_id not in ques_to_idx:
        return np.zeros((1,tot_users))[0]
    
    question_idx = ques_to_idx[question_id]
    
    return user_to_ques[:,question_idx]


if __name__ == '__main__':
    print len(get_ques_feature("d3b63d3e7efcc4c942751c4eddce3638"))
    print len(get_user_feature("7b4f71989c4cefb93a1c639940aa032e")), len(get_user_feature("18ef078c925908094fa5302805a71cac"))
    
    
    