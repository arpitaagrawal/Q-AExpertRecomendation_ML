from collections import Counter
import collections
  
# user id to user map
# ques tag_usertag_Y/N

users = {}
user_word_id = Counter({})
user_char_id = Counter({})
user_tags = Counter({})
# question id to question map
questions = {}
question_word_id = Counter({})
question_char_id = Counter({})
question_tags = Counter({})

userTag_quesTag_dict = dict()
with open("../../bytecup2016data/user_info.txt") as f:
    ques_data = f.readline().strip().split("\t")
    while ques_data and len(ques_data) == 4 :
        users[ques_data[0]] = ques_data
        user_tags.update(ques_data[1].split("/"))
        ques_data = f.readline().strip().split("\t")

# retaining top 500 features
user_word_id = user_word_id.most_common(500)
user_char_id = user_char_id.most_common(500)

print "users", len(users)
print "number of user_tags", len(user_tags)

                   
with open("../../bytecup2016data/question_info.txt") as f:
    question_data = f.readline().strip().split("\t")
    while question_data and len(question_data) == 7 :
        questions[question_data[0]] = question_data
        question_tags.update(question_data[1].split("/"))
        question_data = f.readline().strip().split("\t")
        
print "number_of_question_tags", len(question_tags)
        
        
def main_fn():
    global userTag_quesTag_dict
    global questions
    with open("../../bytecup2016data/invited_info_train.txt") as f:
        training_data = f.readline().strip().split("\t")
        
        #question_tags_dict = {}
        while training_data and len(training_data) == 3 :
            
            question_tags = questions[training_data[0]][1].split("/")
            user_tags = users[training_data[1]][1].split("/")
            question_tags = question_tags
            for ut in user_tags:
                try:
                    for qt in question_tags:
                        if qt not in userTag_quesTag_dict[ut]:
                            userTag_quesTag_dict[ut].append(qt)
                except KeyError:
                    userTag_quesTag_dict[ut] = question_tags
            #print userTag_quesTag_dict
            training_data = f.readline().strip().split("\t") 
    
    #sorted_dict = sorted(userTag_quesTag_dict)
    
    for key in sorted(userTag_quesTag_dict, key=my_key):
        print "%s: %s" % (key, userTag_quesTag_dict[key])

def my_key(dict_key):
    try:
        return int(dict_key)
    except ValueError:
        return dict_key    

main_fn()        
