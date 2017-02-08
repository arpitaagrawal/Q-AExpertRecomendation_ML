'''
compute Cartesian product of user_id and question_id characters and tags
'''

from collections import Counter
from simple_expansion import simple_expansion_feature as simp
import cPickle as pickle
# Each class has it's own Counter to keep track of pair frequency in that class
pairs_class_i = [Counter([]), Counter([])]
list_all_pairs = []
# score for all pairs 
pair_score = {}

def add_pair(pair, class_i):
    '''
    Adds a pair to Counter of class_i tuples
    class_i = 0/1 based on class id
    pair = One pair of character observed in question_id, user_id
    '''
    class_i = int(class_i)
    pairs_class_i[class_i].update([pair])



def get_word_pairs(question_id, user_id):
    '''
    Get all possible word pairs between words in question_id, user_id
    return (user_char,ques_char)
    '''
    user_chars = simp.get_user_words(simp.users[user_id])
    ques_chars = simp.get_question_words(simp.questions[question_id])
    pairs = []
    for user_char in user_chars:
        for ques_char in ques_chars:
            pairs.append((user_char, ques_char))
            
    return pairs

def get_char_pairs(question_id, user_id):
    '''
    Get all possible char pairs between words in question_id, user_id
    '''
    user_chars = simp.get_user_char(simp.users[user_id])
    ques_chars = simp.get_question_char(simp.questions[question_id])
    pairs = []
    for user_char in user_chars:
        for ques_char in ques_chars:
            pairs.append((user_char, ques_char))
            
    return pairs

if __name__ == '__main__':
    pair_features = [("best_words.p",get_word_pairs), ("best_chars.p",get_char_pairs)]
    
    for file_name, get_pairs in pair_features:
        # Each class has it's own Counter to keep track of pair frequency in that class
        pairs_class_i = [Counter([]), Counter([])]
        list_all_pairs = []
        # score for all pairs 
        pair_score = {}

        with open(simp.INVITED_INFO_TRAIN) as f:
            training_data = f.readline().strip().split("\t")
            count = 0
            while training_data and len(training_data) == 3 :
                label = training_data[2]
                question_id = training_data[0]
                user_id = training_data[1]
                
                pairs = get_pairs(question_id, user_id)
                        
                for pair in pairs:
                    add_pair(pair, label)
                    
                training_data = f.readline().strip().split("\t")
                count += 1
                if(count % 10000) == 0:
                    print count, "processed"
                    
                    
                    
        #list_all_pairs is all possible list of pairs found training data
        list_all_pairs = set(pairs_class_i[0].keys() + pairs_class_i[1].keys())
        
        abs_threshold = 0.95
        best_words = set([])
        
        
        for pair in list_all_pairs:
            freq_0 = pairs_class_i[0][pair]
            freq_1 = pairs_class_i[1][pair]
            score = 1.0 * (freq_0 - freq_1) / sum([freq_0,freq_1])
            if abs(score)  > abs_threshold and (freq_0 + freq_1 > 15):
                best_words.add(pair[0])
                best_words.add(pair[1])
            
        best_words.remove("")
        print "len",file_name, len(best_words)
        
        pickle.dump( best_words, open("./feature/" + file_name, "wb"), protocol=2 )
        
        
#         pickle.dump( pair_score, open("./feature/"+file_name, "wb"), protocol=2 )
    

