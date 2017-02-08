import cPickle as pickle

labels = pickle.load( open("../feature_engg/feature/labels.p", "rb") )

features = pickle.load( open("../feature_engg/feature/all_features.p", "rb") )

print "data loaded", labels[0]

count_1 = 0
count_0 = 0
data_0 = []
data_1 = []

for idx,label in enumerate(labels):
    if label == "0" and count_0 <21:
        data_0.append(features[idx][0:44])
        count_0+=1
    if label == "1" and count_1 <21:
        data_1.append(features[idx][0:44])
        count_1+=1
        
    if count_0 + count_1 >=40:
        break
        
print "1"
for data in data_1: print data

print ""
print ""
print "0"
for data in data_0: print data

