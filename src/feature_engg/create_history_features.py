from simple_expansion import simple_expansion_feature as simp
from collections import Counter
import numpy as np
import random
from clusters.get_distance_metric_col import get_distance_metric_for_user
from clusters.get_ques_similarity import get_distance_metric_for_ques
import time

MIN_NUM = 100
user_user_sim_cache_L1 = {}
user_user_sim_cache_L2 = {}
ques_ques_sim_cache_L1 = {}
ques_ques_sim_cache_L2 = {}

def pair_available_dict(p1, p2, cache):
    '''
    return cache[p1][p2] or cache[p2][p1] if available else None
    '''
    if p1 in cache:
        if p2 in cache[p1]:
            return cache[p1][p2]
    if p2 in cache:
        if p1 in cache[p2]:
            return cache[p2][p1]
        
    return None 

def set_pair(p1, p2, cache,value):
    '''
    set cache[p1][p2]=value
    '''
    if not p1 in cache:
        cache[p1] = {}
    
    cache[p1][p2] = value
    
            
class UserHistory:
    'Class to store history based features for all users'
    question_ans = []
    question_not_ans = []
    def __init__(self):
        self.question_ans = []
        self.question_not_ans = []
        
    def add_ques_ans(self, ques):
        self.question_ans.append(ques)
    
    def add_ques_not_ans(self, ques):
        self.question_not_ans.append(ques)
    
    ## feature array format
    ## 0 -> num of question_id ans
    ## 1 -> num of question_id NOT ans
    ## 2 -> num of time similar tags question_id ans
    ## 3 -> num of time unsimilar tags question_id ans
    ## 4 -> num of time similar tags question_id NOT ans
    ## 5 -> num of time unsimilar tags question_id NOT ans

    def get_feature_train(self, ques_asked):
        #EXCLUDE TRAINING QUESTION BEFORE CALCULATING FEATURE
        removed_ans = False
        removed_not_ans = False
        
        if(ques_asked in self.question_ans):
            self.question_ans.remove(ques_asked)
            removed_ans = True
#         else: for unknown data
        if(ques_asked in self.question_not_ans):
            self.question_not_ans.remove(ques_asked)
            removed_not_ans = True
        
        # 0,1
        feature = [len(self.question_ans),len(self.question_not_ans) ]
        q_tag = simp.get_question_tag(simp.questions[ques_asked])[0]
        
        ans_tags = Counter([])
        for ques in self.question_ans:

            ans_tags.update(simp.get_question_tag(simp.questions[ques]))
          
        not_ans_tags = Counter([])
        for ques in self.question_not_ans:
            not_ans_tags.update(simp.get_question_tag(simp.questions[ques]))
        
        
        # 2,3
        feature.append(ans_tags[q_tag])
        ans_tags[q_tag] = 0
        feature.append(sum(ans_tags.values()))
        
        # 4,5
        feature.append(not_ans_tags[q_tag])
        not_ans_tags[q_tag] = 0
        feature.append(sum(not_ans_tags.values()))

        # 6,7
        # TODO: do the same for SD, max, min
        ham_dist = [0]
        L2_dist = [0]
        min_num = MIN_NUM
        
        if len(self.question_ans) < min_num: min_num = len(self.question_ans) 
        question_ans_rand = random.sample(set(self.question_ans), min_num)
        for ques in question_ans_rand:
            L2_dist.append(get_distance_metric_for_ques(ques, ques_asked, metric="l2"))
            ham_dist_i = get_distance_metric_for_ques(ques, ques_asked, metric="hamming")
            if ham_dist_i:
                ham_dist.append(ham_dist_i)
          
        # if len(self.question_ans) = 0 give a neg distance 
        if (len(self.question_ans) == 0): 
            feature.append(0.0005)
            feature.append(0.0005)
            feature.append(0.0005)
            feature.append(0)
            
            feature.append(500)
            feature.append(500)
            feature.append(500)
            feature.append(0)
            
        else:
            feature.append(max(ham_dist))
            feature.append(min(ham_dist))
            feature.append(np.mean(ham_dist))
            feature.append(np.std(ham_dist))
            
            feature.append(max(L2_dist))
            feature.append(min(L2_dist))
            feature.append(np.mean(L2_dist))
            feature.append(np.std(L2_dist))
              
        ham_dist = [0]
        L2_dist = [0]
        min_num = MIN_NUM
          
        if len(self.question_not_ans) < min_num: min_num = len(self.question_not_ans) 
        question_not_ans_rand = random.sample(set(self.question_not_ans), min_num)
        for ques in question_not_ans_rand:
            L2_dist.append(get_distance_metric_for_ques(ques, ques_asked, metric="l2"))
            ham_dist_i = get_distance_metric_for_ques(ques, ques_asked, metric="hamming")
            if ham_dist_i:
                ham_dist.append(ham_dist_i)
           
        # if len(self.question_not_ans) = 0 give a lower distance
        if(len(self.question_not_ans) == 0):
            feature.append(0)
            feature.append(0)
            feature.append(0)
            feature.append(0)
            
            feature.append(0)
            feature.append(0)
            feature.append(0)
            feature.append(0)
            
        else: 
            feature.append(max(ham_dist))
            feature.append(min(ham_dist))
            feature.append(np.mean(ham_dist))
            feature.append(np.std(ham_dist))
            
            feature.append(max(L2_dist))
            feature.append(min(L2_dist))
            feature.append(np.mean(L2_dist))
            feature.append(np.std(L2_dist))
        
        if removed_ans:
            self.question_ans.append(ques_asked)
        if removed_not_ans:
            self.question_not_ans.append(ques_asked)
            
            
        return feature

class QuesHistory:
    'Class to store not history based features for all question_id'
    user_ans = [] ##IDs of users who've answered this question_id
    user_not_ans = [] ##
    def __init__(self):
        self.user_ans = []
        self.user_not_ans = []

    def add_user_ans(self, user_id):
        '''

        :param user_id: user_id ID
        :return:
        '''
        self.user_ans.append(user_id)
    
    def add_user_not_ans(self, user_id):
        '''

        :param user_id: user_id ID
        :return:
        '''
        self.user_not_ans.append(user_id)
        
    ## feature array format
    ## 0 -> num of user_id ans
    ## 1 -> num of user_id not ans
    ## 2 -> num of time similar tag of user_id ans
    ## 3 -> num of time unsimilar tag of user_id ans
    ## 4 -> num of time similar tag of user_id NOT ans
    ## 5 -> num of time unsimilar tag of user_id NOT ans

    def get_feature_train(self, user_target):
        '''

        :param user_target: user_id ID
        :return: all features based on this question_id and user_id ID
        '''
        removed_ans = False
        removed_not_ans = False
        
        if(user_target in self.user_ans):
            self.user_ans.remove(user_target)
            removed_ans = True
#         else:
        if(user_target in self.user_not_ans):
            self.user_not_ans.remove(user_target)
            removed_not_ans = True
            
        feature = [len(self.user_ans),len(self.user_not_ans)]
        
        u_tag = simp.get_user_tag(simp.users[user_target])
        
        ans_tags = Counter([])
        for user_id in self.user_ans:
            ans_tags.update(simp.get_user_tag(simp.users[user_id]))
                 
        not_ans_tags = Counter([])
        for user_id in self.user_not_ans:
            not_ans_tags.update(simp.get_user_tag(simp.users[user_id]))
        
        
        # 2,3
        sim_tag_ans = 0
        for tag in u_tag:
            sim_tag_ans+=ans_tags[tag]
            ans_tags[tag] = 0
            
        feature.append(sim_tag_ans)
        feature.append(sum(ans_tags.values()))
        
        # 4,5
        not_sim_tag_ans = 0
        for tag in u_tag:
            not_sim_tag_ans+=not_ans_tags[tag]
            not_ans_tags[tag] = 0
        
        feature.append(not_sim_tag_ans)
        feature.append(sum(not_ans_tags.values()))    
        
        # 6,7,8,9 = max, min, std, mean L2 
        # 10,11,12,13 = max, min, std, mean L2
        ham_dist = [0]
        L2_dist = [0]
        min_num = MIN_NUM
        if len(self.user_ans) < min_num: min_num = len(self.user_ans) 
        user_ans_rand = random.sample(set(self.user_ans), min_num)
          
        for user_id in user_ans_rand:
            L2_dist.append(get_distance_metric_for_user(user_target, user_id, metric="l2"))
            ham_dist_i = get_distance_metric_for_user(user_target, user_id, metric="hamming")
            if ham_dist_i:
                ham_dist.append(ham_dist_i)
                
        if(len(self.user_ans) == 0):
            feature.append(0.005)
            feature.append(0.005)
            feature.append(0.005)
            feature.append(0)
            
            feature.append(5)
            feature.append(5)
            feature.append(5)
            feature.append(0)
            
        else:
            feature.append(max(ham_dist))
            feature.append(min(ham_dist))
            feature.append(np.mean(ham_dist))
            feature.append(np.std(ham_dist))
            
            feature.append(max(L2_dist))
            feature.append(min(L2_dist))
            feature.append(np.mean(L2_dist))
            feature.append(np.std(L2_dist))
            
         
        ham_dist = [0]
        L2_dist = [0]
        min_num = MIN_NUM
        if len(self.user_not_ans) < min_num: min_num = len(self.user_not_ans) 
        user_not_ans_rand = random.sample(set(self.user_not_ans), min_num)
         
        for user_id in user_not_ans_rand:
            L2_dist.append(get_distance_metric_for_user(user_target, user_id, metric="l2"))
            ham_dist_i = get_distance_metric_for_user(user_target, user_id, metric="hamming")
            if ham_dist_i:
                ham_dist.append(ham_dist_i)
 
        if(len(self.user_not_ans) == 0):
            feature.append(0)
            feature.append(0)
            feature.append(0)
            feature.append(0)
            
            feature.append(0)
            feature.append(0)
            feature.append(0)
            feature.append(0)
            
            
        else:
            feature.append(max(ham_dist))
            feature.append(min(ham_dist))
            feature.append(np.mean(ham_dist))
            feature.append(np.std(ham_dist))
            
            feature.append(max(L2_dist))
            feature.append(min(L2_dist))
            feature.append(np.mean(L2_dist))
            feature.append(np.std(L2_dist))
         
        if removed_ans:
            self.user_ans.append(user_target)
        if removed_not_ans:
            self.user_not_ans.append(user_target)
            
        return feature


            
def get_history_feature():
    '''
    return dictionary<user_id,feature_array>
    We calculate feature for all users at once and return dictionary<user_id,feature_array> 
    which can be referenced multiple times
    '''
    
    ## Dictionary<user_id, UserHistory> to store feature array for all user_id   
    user_to_feature = {}
    ## Dictionary<ques_id, QuesHistory> to store feature array for all questions   
    ques_to_feature = {}
    
    with open(simp.INVITED_INFO_TRAIN) as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            label = training_data[2]
            question_id = training_data[0]
            user_id = training_data[1]
            #fill dict if empty
            if user_id not in user_to_feature: user_to_feature[user_id] = UserHistory() 
            if question_id not in ques_to_feature: ques_to_feature[question_id] = QuesHistory()
            
            if label == '1' :
                user_to_feature[user_id].add_ques_ans(question_id)
                ques_to_feature[question_id].add_user_ans(user_id)
            if label == '0' :
                user_to_feature[user_id].add_ques_not_ans(question_id)
                ques_to_feature[question_id].add_user_not_ans(user_id)
                
            training_data = f.readline().strip().split("\t")
        
    
    return user_to_feature,ques_to_feature

user_to_feature,ques_to_feature = get_history_feature()

print "Number of users and questions in training data", len(user_to_feature) ,len(ques_to_feature)

def get_consolidated_feature_train(question_id, user_id):
    '''
    Return history feature for given user_id, question_id, label. History is calculated excluding current label.
    '''
    # simplify objects to feature
    if user_id in user_to_feature:
        user_f = user_to_feature[user_id].get_feature_train(question_id)
    else :
        print "unknown user"
        user_f = [0] * 22
    if question_id in ques_to_feature:
        question_f = ques_to_feature[question_id].get_feature_train(user_id)
    else :
        print "unknown question"
        question_f = [0] * 22
        
    return user_f + question_f

if __name__ == '__main__':
    print "Start",time.time()
    print "Feature length",len(get_consolidated_feature_train("d3b63d3e7efcc4c942751c4eddce3638", "18ef078c925908094fa5302805a71cac"))
    print "Feature length",len(get_consolidated_feature_train("d3b63d3e7efcc4c942751c4eddce3638", "7b4f71989c4cefb93a1c639940aa032e"))
    qh = QuesHistory()
    qh.add_user_ans("7b4f71989c4cefb93a1c639940aa032e")

    print qh.get_feature_train("7b4f71989c4cefb93a1c639940aa032e")
    print "End",time.time()

    