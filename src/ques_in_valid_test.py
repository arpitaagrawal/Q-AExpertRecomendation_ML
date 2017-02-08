import cPickle as pickle 

ques_va_te = set([])

## validate_nolabel test_nolabel

def add_all_ques(fname):
    with open(fname,"r") as f:
        f.readline().strip().split(",")
        training_data = f.readline().strip().split(",")
        while training_data and len(training_data) >= 2 :
            ques_id = training_data[0]
            ques_va_te.add(ques_id)
            training_data = f.readline().strip().split(",")
            
    

if __name__ == '__main__':
    add_all_ques("../bytecup2016data/test_nolabel.txt")
    add_all_ques("../bytecup2016data/validate_nolabel.txt")
    print len(ques_va_te)
    pickle.dump(list(ques_va_te) , open("../bytecup2016data/ques_va_te.p","wb"), protocol=2)