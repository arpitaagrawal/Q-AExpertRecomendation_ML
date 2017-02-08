'''
Created on 19-Nov-2016

@author: madhav
'''
import numpy as np

zero_pair = set([])
with open("../bytecup2016data/invited_info_train_class1.txt", "r") as f:
    test_data = f.readline().strip().split("\t")
    while test_data and len(test_data) >= 2 :
        question_id = test_data[0]
        user_id = test_data[1]
        
        zero_pair.add((question_id,user_id))
        
        test_data = f.readline().strip().split("\t")
        
print len(zero_pair)

files = ["../bytecup2016data/test_label_mahout.csv","../bytecup2016data/test_label.csv"]

files_open = [open(fi,"r") for fi in files]

count = 0
with open("../bytecup2016data/test_label_final.csv", "w") as fo:
    fo.write(files_open[0].readline())
    [f.readline() for f in files_open[1:]]
    lineArr = [f.readline().strip().split(",") for f in files_open]
    
    while len(lineArr[0]) == 3:
        
        all_pred = [ float(l[2]) for l in lineArr] 
        all_pred.append(np.random.normal(0,0.1))
        all_pred = np.array(all_pred)
        
#         print lineArr[0][0] , lineArr[0][1], all_pred, sum(all_pred)/len(all_pred)
        fo.write(lineArr[0][0] + "," + lineArr[0][1] + "," + str(np.average(all_pred)) )
        
        fo.write("\n")
        
        lineArr = [f.readline().strip().split(",") for f in files_open]
        

print count