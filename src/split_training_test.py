
from sklearn.cross_validation import train_test_split

file_name = "../bytecup2016data/invited_info_train_PROC.txt"
tr_file_name = "../bytecup2016data/invited_info_train_PROC_tr.txt"
test_file_name = "../bytecup2016data/invited_info_train_PROC_test.txt"


if __name__ == '__main__':
    with open(file_name) as f:
        data = f.readlines()
        
    train, test = train_test_split(data, train_size = 0.85)
    
    with open(tr_file_name,"w") as f:
        for item in train:
            f.write("%s" % item)
    
    with open(test_file_name,"w") as f:
        f.write("qid,uid,label\n")
        for item in test:
            item = ",".join(item.split("\t"))
            f.write("%s" % item)