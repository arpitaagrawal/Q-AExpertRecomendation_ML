import graphlab as gl


user_arr = []
item_arr = []
rate_arr = []

INVITED_INFO_TRAIN = "../../bytecup2016data/invited_info_train.txt"

with open(INVITED_INFO_TRAIN) as f:
    training_data = f.readline().strip().split("\t")
    while training_data and len(training_data) == 3 :
        
        label = training_data[2]
        
        question = training_data[0]
        user = training_data[1]
        
#         if label == "1":
        rate_arr.append(int(label) )  
        user_arr.append(user)
        item_arr.append(question)
        
        training_data = f.readline().strip().split("\t")
        
print len(user_arr)

user_arr_t = []
item_arr_t = []
label_arr_t = []

# test_nolabel.txt validate_nolabel.txt invited_info_train_PROC_test
with open("../../bytecup2016data/test_nolabel.txt","r") as f:
    training_data = f.readline().strip().split(",")
    training_data = f.readline().strip().split(",")
    print training_data
    while training_data and len(training_data) >= 2 :
        question = training_data[0]
        user = training_data[1]
        
        user_arr_t.append(user)
        item_arr_t.append(question)
    
        training_data = f.readline().strip().split(",")

      
sf = gl.SFrame({'user_id':  user_arr, 'item_id': item_arr, 'target': rate_arr })

# train, test = gl.recommender.util.random_split_by_user(sf)

train = sf

# |   1    | 0.0386473429952 | 0.0236178207193 |  0.001
# |   1    | 0.0531400966184 | 0.0271604938272 |  0.1
# |   1    | 0.0458937198068 | 0.0259891879457 |  1.0


# m3 = gl.ranking_factorization_recommender.create(train, ranking_regularization = 1, unobserved_rating_value = 0.4, verbose=False)

test = gl.SFrame({'user_id':  user_arr_t, 'item_id': item_arr_t})

m3 = gl.ranking_factorization_recommender.create(train, target='target')
# m3 = gl.ranking_factorization_recommender.create(train, target='target', binary_target=True)

# max_iter = 50, 36.0158114388
# DEF 35.9802318094



label_arr_t = m3.predict(test)
min_val = min(label_arr_t)
max_val = max(label_arr_t)
 
#test_label_graphlab validate_label_graphlab
with open("../../bytecup2016data/test_graph_pos.csv", "w") as f:
    f.write("qid,uid,label\n")
    for user, item, label in zip(user_arr_t, item_arr_t, label_arr_t):
        label = (label - min_val) / (max_val - min_val)
        f.write(item + "," + user + "," + str(label) + "\n" )

# report = m3.evaluate_precision_recall(test, exclude_known = True, verbose=False)

# report = m3.evaluate(test, exclude_known_for_precision_recall=True, verbose=False)

# print report["precision_recall_overall"]
