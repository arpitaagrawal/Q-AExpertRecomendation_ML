'''
Dumps flattened features in feature directory in below format
Q_1F_1    Q_1F_2    Q_1F_3    U_1F_1    U_1F_2    U_1F_3    U_1F_4    0
Q_2F_1    Q_2F_2    Q_2F_3    U_2F_1    U_2F_2    U_2F_3    U_2F_4    1
..
..
Q_nF_1    Q_nF_2    Q_nF_3    U_nF_1    U_nF_2    U_nF_3    U_nF_4    0

'''
from collections import Counter
import cPickle as pickle
from tempfile import TemporaryFile
import numpy as np
import matplotlib.pyplot as plt

  
# user id to user map
users = {}
user_word_id = Counter({})
user_char_id = Counter({})
user_tags = Counter({})
# question id to question map
questions = {}
question_word_id = Counter({})
question_char_id = Counter({})
question_tags = Counter({})

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
print "user_tags", len(user_tags)
                   
with open("../../bytecup2016data/question_info.txt") as f:
    question_data = f.readline().strip().split("\t")
    while question_data and len(question_data) == 7 :
        questions[question_data[0]] = question_data
        question_tags.update(question_data[1].split("/"))
        question_data = f.readline().strip().split("\t")


print "questions", len(questions)
print "question_tags", len(question_tags)

def get_one_feature(item_set, global_set):
    '''
    item_set - COUNTER of values present in current item
    global_set - COUNTER of values present in whole training set
    return - a feature of length global_set with value set if feature is present in item_set else 0
    '''
    feature_one = []
    for tag in global_set :
        feature_one.append(item_set[tag])
    
    return feature_one

def get_full_feature(question, user):
    feature = []
    
    #Number of upvotes
    feature.append(int(question[4]) )
    ##Number of answers
    feature.append(int(question[5]) )
    ## Fill #top quality answers
    feature.append(int(question[6]) )
    
    # # fill features with user vector
    #feature.extend(get_one_feature(Counter(user[1].split("/")), user_tags))    
    return feature

def plot_scatter_graph(features, labels):
    

    label_arr = ["Number of upvotes","Number of answers","Number of top quality answers"]
    for key in []:
        plot_grph(features[:,key], labels,label_arr[key], " Output label, If the ques was answered" )
    #graph between num_of_votes and no_of_answers
#     print'num of upvotes', (features[0:100,0])
#     print'num of answers', (features[0:100,1])
#     plot_grph(features[:,0], features[:,1],label_arr[0], label_arr[1] )
    
    #graph between top quality answers and no_of_answers
    print'#top quality answers', (features[0:100,2])
    print'num of answers', (features[0:100,1])
    plot_grph(features[:,2], features[:,1],label_arr[2], label_arr[1] )
    
    #graph between num_of_votes and top quality answers
#     print'#top quality answers', (features[0:100,2])
#     print'num of upvotes', (features[0:100,0])
# #
#     plot_grph(features[:,0], features[:,2],label_arr[0], label_arr[2] )


def plot_grph(x,y, label_x, label_y):
    
#     plt.axis([0.0,2500.0, 0,200.0])
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

    plt.axis([0.0,50.0, 0,50.0])
    ax = plt.gca()
    ax.set_autoscale_on(False)

#     plt.axis([0.0,500.0, 0,80.0])
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
    plt.figure(1)
    plt.subplots_adjust(hspace=1)

    plt.plot(x, y, "o")
    plt.ylabel(label_y)
    plt.xlabel(label_x)

    plt.show()    

def main_fn():
    labels = []
    features = []
    count_nzero = 0
    with open("../../bytecup2016data/invited_info_train.txt") as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            if int(training_data[2]) >= 1:
                #print training_data[2]
                count_nzero += 1
            labels.append(training_data[2])
            
            question = questions[training_data[0]]
            user = users[training_data[1]]
            features.append(get_full_feature(question, user))                
            training_data = f.readline().strip().split("\t")

    #print "features", len(features)
    ##print np.unique(labels)
    #print "labels", len(labels)
#     count_nzero = np.count_nonzero(labels)
    print "non zero labels", count_nzero
    print "zero labels", (245752- count_nzero)
    
    
    
    fact = np.arange(0.00000366222, 0.9, 0.00000366222)
    plot_scatter_graph(np.array(features), np.array(labels).astype(float) + fact)
    np.savetxt('test.out', np.array(features), delimiter=',') 
    
main_fn()