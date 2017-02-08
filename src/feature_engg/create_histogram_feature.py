import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

print "Run store_histogram_feature in case not done already"

def get_best_histogram(pair_score):  
    return np.histogram(pair_score.values(), bins="auto") 


pair_score_char = pickle.load(open("../feature_engg/feature/distinguish_char.p", "r"))
hist_char, bins_char = get_best_histogram(pair_score_char)

pair_score_tag = pickle.load(open("../feature_engg/feature/distinguish_tag.p", "r"))
hist_tag, bins_tag = get_best_histogram(pair_score_tag)

pair_score_word = pickle.load(open("../feature_engg/feature/distinguish_word.p", "r"))
hist_word, bins_word = np.histogram(pair_score_word.values(), bins=60)

print "Created all histograms "

def get_feature(pair_list, feature="char"):
    '''
    @param pair_list: List of cross product features between user and question
    @param feature: feature = char/tag
     
    Given list of pair calculates a vector by using histogram
    '''
    pair_score = []
    bins = []
    if feature == "char":
        pair_score = pair_score_char
        bins = bins_char
    elif feature == "tag":
        pair_score = pair_score_tag
        bins = bins_tag
    elif feature == "word":
        pair_score = pair_score_word
        bins = bins_word
        
    pair_list_score = []
    for pair in pair_list:
        if pair in pair_score:
            pair_list_score.append(pair_score[pair])
        
    set_indices = np.digitize(pair_list_score, bins)
    feature_vec = [0]*len(bins)
    for set_index in set_indices:
        
        feature_vec[set_index-1] += 1.0
    
    return feature_vec
    
def plot_histogram(x, var_name, bins=10):
    '''
    plots histogram of x array, 10 bins equal size
    '''
    _, hist_data, _ = plt.hist(x, bins=bins)
    print type(hist_data), hist_data
    plt.plot(x=hist_data)
    plt.savefig(var_name, linewidth=0)
    plt.close()

def plot_best_histogram(pair_score):
    hist, bins = get_best_histogram(pair_score)
    
    print hist, bins
    
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    
    plt.savefig("best-freq_0-freq_1", linewidth=0)
    plt.close()
    
    

if __name__ == '__main__':
#     for bins in range(0,151,10):
#         if bins == 0:
#             bins = 10
#         plot_histogram(pair_score_word.values(), str(bins) + "-freq_0-freq_1-word", bins)
#     
#     plot_best_histogram(pair_score_tag)
    
    print list(np.digitize([-1,-1,-0.5,0,0.5,1], bins_tag))
    print list(np.digitize([-1,-1,-0.5,0,0.5,1], bins_char))
    
    print get_feature([('1133', '855'), ('770', '1772'), ('91', '14'), ('52', '855'), ('560', '55'), ('142', '511'), ('135', '855'), ('141', '2072'), ('619', '512'), ('1566', '512'), ('619', '2072'), ('52', '511'), ('770', '56'), ('52', '73'), ('141', '14'), ('1039', '512'), ('560', '512'), ('1133', '73'), ('939', '1772'), ('1194', '1772'), ('1566', '14'), ('62', '55'), ('1002', '14'), ('560', '2072'), ('1039', '55'), ('560', '630'), ('1002', '511'), ('52', '2072'), ('141', '511'), ('62', '14'), ('52', '630'), ('135', '1772'), ('135', '55'), ('141', '55'), ('1194', '630'), ('224', '630'), ('135', '73'), ('1039', '73'), ('1133', '56'), ('619', '56'), ('1566', '56'), ('52', '56'), ('73', '1772'), ('1039', '855'), ('939', '56'), ('1194', '511'), ('1002', '73'), ('619', '855'), ('1133', '55'), ('224', '14'), ('1566', '2072'), ('91', '630'), ('135', '630'), ('1133', '2072'), ('142', '1772'), ('1567', '55'), ('1567', '1772'), ('1194', '2072'), ('73', '855'), ('141', '1772'), ('770', '511'), ('1194', '512'), ('1133', '1772'), ('770', '855'), ('91', '56'), ('1566', '511'), ('52', '512'), ('62', '1772'), ('1194', '55'), ('770', '512'), ('91', '855'), ('52', '14'), ('142', '855'), ('1039', '56'), ('939', '855'), ('1567', '855'), ('1194', '855'), ('91', '55'), ('73', '2072'), ('141', '855'), ('141', '56'), ('62', '73'), ('1194', '56'), ('1566', '855'), ('770', '14'), ('135', '512'), ('91', '512'), ('1133', '630'), ('1566', '55'), ('73', '56'), ('91', '2072'), ('224', '55'), ('939', '55'), ('1133', '512'), ('1039', '14'), ('1567', '630'), ('142', '630'), ('141', '73'), ('224', '73'), ('224', '511'), ('1039', '1772'), ('939', '14'), ('939', '512'), ('560', '855'), ('91', '511'), ('73', '55'), ('619', '1772'), ('1002', '512'), ('1566', '1772'), ('142', '56'), ('1567', '56'), ('1133', '511'), ('560', '1772'), ('62', '2072'), ('224', '2072'), ('62', '512'), ('135', '14'), ('770', '2072'), ('224', '855'), ('1567', '511'), ('1567', '73'), ('1002', '2072'), ('142', '73'), ('560', '73'), ('73', '512'), ('91', '73'), ('939', '630'), ('1002', '55'), ('142', '55'), ('560', '14'), ('142', '2072'), ('224', '56'), ('1567', '2072'), ('73', '14'), ('62', '855'), ('619', '511'), ('135', '2072'), ('1566', '630'), ('91', '1772'), ('73', '511'), ('619', '630'), ('73', '73'), ('1002', '56'), ('1039', '511'), ('224', '1772'), ('1002', '1772'), ('1194', '73'), ('62', '56'), ('939', '73'), ('135', '511'), ('939', '2072'), ('1002', '855'), ('142', '14'), ('1566', '73'), ('1567', '14'), ('619', '73'), ('1194', '14'), ('1002', '630'), ('73', '630'), ('1039', '2072'), ('770', '630'), ('135', '56'), ('1133', '14'), ('619', '14'), ('62', '630'), ('560', '511'), ('224', '512'), ('939', '511'), ('619', '55'), ('52', '55'), ('52', '1772'), ('141', '630'), ('62', '511'), ('560', '56'), ('142', '512'), ('1567', '512'), ('770', '73'), ('141', '512'), ('1039', '630'), ('770', '55')])
    print get_feature([('1', '1'), ('1', '2')], feature="tag")
    print get_feature([('3599', '2129')], feature="word")

