from collections import Counter

def print_user_answered():
    users_answered = Counter({})
    questions_answered = Counter({})
    with open("../bytecup2016data/invited_info_train_class1.txt") as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            label = training_data[2]
            if label == '1':
                question_id = training_data[0]
                user_id = training_data[1]
                users_answered.update([user_id])
                questions_answered.update([question_id])
            
            training_data = f.readline().strip().split("\t")
            
#     for user_id in users_answered.most_common():
#         print user_id[0],  user_id[1]   
#     
    
    for question_id in questions_answered.most_common():
        print question_id[0],  question_id[1]   
            
        

if __name__ == '__main__':
    print_user_answered()