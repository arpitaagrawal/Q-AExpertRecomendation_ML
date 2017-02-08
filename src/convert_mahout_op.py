import cPickle

def convert_to_biendata_format():
    '''
    Convert ques_id, user_int_id to user int id, ques int id
    stores a mapping of id to int id
    '''
    bien_file = open("../bytecup2016data/validate_label.csv","w")
    bien_file.write("qid,uid,label\n")
    
    user_to_idx = cPickle.load(open("../bytecup2016data/user_to_idx.p","rb"))
    ques_to_idx = cPickle.load(open("../bytecup2016data/ques_to_idx.p","rb"))

    idx_to_user = {str(v): k for k, v in user_to_idx.iteritems()}
    idx_to_ques =  {str(v): k for k, v in ques_to_idx.iteritems()}
    
    with open("../bytecup2016data/validate_label_mahout.csv","r") as f:
        test_data = f.readline().strip().split(",")
        
        while test_data and len(test_data) == 3 :
            user_int_id = test_data[0]
            question_int_id = test_data[1]
            label = test_data[2]
              
            bien_file.write( "{0},{1},{2}\n".format(idx_to_ques[question_int_id], idx_to_user[user_int_id], label) )
                            
            test_data = f.readline().strip().split(",")
        f.close()
        bien_file.close()
        
        
if __name__ == '__main__':
    convert_to_biendata_format()
    print "Finished writing output file"