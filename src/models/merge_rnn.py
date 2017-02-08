import numpy as np
from simple_expansion import simple_expansion_feature as simp 

ques_feat = []
labels = []

with open("../../bytecup2016data/invited_info_train_test.txt") as f:
    training_data = f.readline().strip().split(",")
    training_data = f.readline().strip().split(",")
    while len(training_data) == 3:
        labels.append(training_data[2])
            
        question = simp.questions[training_data[0]]
        
        feature = []
        ## Fill #upvotes
        feature.append(int(question[4]))
        ## Fill #answers
        feature.append(int(question[5]))
        ## Fill #top quality answers
        feature.append(int(question[6]))
        
        ques_feat.append(feature)
        
        training_data = f.readline().strip().split(",")

#normalize feature between 0 and 1
ques_feat = np.array(ques_feat, dtype="float")
print "ques_feat.shape, ques_feat[0]", ques_feat.shape, ques_feat[0]
for col_i in range(len(ques_feat[0])):
    max_col =  max(ques_feat[:,col_i])
    min_col =  min(ques_feat[:,col_i])
    
    new_col =  [ (1.0*(x - min_col))/(max_col - min_col) for x in ques_feat[:,col_i]]
    ques_feat[:,col_i] = new_col
    

print "ques_feat.shape, ques_feat[0]", ques_feat.shape, ques_feat[0]

#testing line
# labels = labels[:18]

labels = np.array(labels)

files = [  "../../bytecup2016data/rnn_tr_word.csv","../../bytecup2016data/rnn_tr_char.csv",
           "../../bytecup2016data/rnn_tr_tag.csv" , 
         #"../../bytecup2016data/train_test_nn.csv",
         # "../../bytecup2016data/graph_tr.csv"
         ]
#Add graph lab

files_open = [open(fi,"r") for fi in files]

#     fo.write(files_open[0].readline())
#     [f.readline() for f in files_open[1:]]
#     lineArr = [f.readline().strip().split(",") for f in files_open]

lineArr = [f.readline().strip().split(",") for f in files_open]
all_models = []
for i in range(len(labels)):
    lineArr = [f.readline().strip().split(",") for f in files_open]
    final_arr = []
#     final_arr.extend(ques_feat[i])
    all_pred = np.array([ float(l[2]) for l in lineArr] ) 
    final_arr.extend(all_pred)
    
    all_models.append(final_arr)
    
all_models = np.array(all_models)
print "all_models.shape", all_models.shape

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=1)
clf3 = GaussianNB()
    

train_idx = range(0, int(0.8 * len(labels)) )
test_idx = range(len(train_idx), len(labels) )

print "len(train_idx), len(test_idx)", len(train_idx), len(test_idx)

X = all_models[train_idx]
y = labels[train_idx]  

eclf3 = VotingClassifier(estimators=[
       ('lr', clf1), ('gnb', clf3)],
       voting='soft', weights=[1,1])

eclf3 = eclf3.fit(X, y)

print test_idx[0], test_idx[-1]
proba = eclf3.predict_proba(all_models[test_idx])

print len(proba), len(labels[test_idx])
print proba, labels[test_idx]


fo = open("../../bytecup2016data/invited_info_train_test_un.txt","w")

with open("../../bytecup2016data/invited_info_train_test.txt") as f:
    training_data = f.readline().strip().split(",")
    training_data = f.readline().strip().split(",")
    count = 0
    while len(training_data) == 3:
        
        if count >= test_idx[0]:     
            prob = proba [count - test_idx[0]][1]
            fo.write(training_data[0] + "," + training_data[1] + "," + format(prob, '.8f') + "\n")
        
        count += 1
        
        
        training_data = f.readline().strip().split(",")



