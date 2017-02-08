'''
max char_features 123
max word_feature 52
max tag_features 8

chosen best_words 5005
chosen best_chars 2488
total unique tags 143
user_tags 143 question_tags 20
'''
from collections import Counter
import cPickle as pickle

from simple_expansion import simple_expansion_feature as simp 

  
def main_fn():
    labels = []
    char_features = []
    word_features = []
    tag_features = []
    
    best_words = pickle.load(open("./feature/best_words.p", "r"))
    best_chars = pickle.load(open("./feature/best_chars.p", "r"))
    
    best_words_dict = dict([ (w,idx) for idx,w in enumerate(best_words)])
    best_chars_dict = dict([ (c,idx) for idx,c in enumerate(best_chars)])
    
    print "best_words", len(best_words)
    print "best_chars", len(best_chars)
    
    with open("../../bytecup2016data/invited_info_train_tr.txt") as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            
            labels.append(training_data[2])
            
            question = simp.questions[training_data[0]]
            user = simp.users[training_data[1]]
            
            char_feature = simp.get_user_char(user)
            char_feature.extend(simp.get_question_char(question))
            # REMOVE LESS FREQUENT ITEMS
            char_feature = [best_chars_dict[c] for c in char_feature if c in best_chars and c != '' ]
            char_features.append(char_feature)
            
            word_feature = simp.get_user_words(user)
            word_feature.extend(simp.get_question_words(question))
            # REMOVE LESS FREQUENT ITEMS
            word_feature = [best_words_dict[w] for w in word_feature if w in best_words and w != '' ]
            word_features.append(word_feature)
            
            tag_feature = simp.get_user_tag(user)
            tag_feature.extend(simp.get_question_tag(question))
            tag_features.append([int(t) for t in tag_feature])
            
            if len(char_features) % 1000 == 0:
                print len(char_features)
                
            training_data = f.readline().strip().split("\t")
            
            
        
    print "features", len(char_features)
    print "labels", len(labels)
        
    pickle.dump(char_features, open("./feature/char_features.p", "wb") )
    pickle.dump(word_features, open("./feature/word_features.p", "wb") )
    pickle.dump(tag_features, open("./feature/tag_features.p", "wb") )
    pickle.dump(labels, open("./feature/labels.p", "wb") )
    
    print "max char_features", max([len(x) for x in char_features]) 
    print "max word_feature", max([len(x) for x in word_features]) 
    print "max tag_features", max([len(x) for x in tag_features]) 
    
#     max char_features 123
#     max word_feature 52
#     max tag_features 8
    print "done"    
    
if __name__ == "__main__":
    # stuff only to run when not called via 'import' here
    main_fn() 