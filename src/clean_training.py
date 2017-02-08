

def remove_duplicate_pair():
    '''
    Strategy - Try to keep the pair with value 1  
    '''
    valid_pair = {}
    with open("../bytecup2016data/invited_info_train.txt","r") as f:
        training_data = f.readline().strip().split("\t")
        while training_data and len(training_data) == 3 :
            question_id = training_data[0]
            user_id = training_data[1]
            pair = (question_id,user_id)
            label = training_data[2]
            if pair in valid_pair:
                if label == '1' :
                    valid_pair[pair]=label
            else:
                valid_pair[pair]=label
                
            training_data = f.readline().strip().split("\t")
        f.close()
    
    with open("../bytecup2016data/invited_info_train_PROC.txt","w") as f:
        for key in valid_pair:
            f.write(key[0] + "\t" + key[1] + "\t" + valid_pair[key] +"\n")
            
        f.close()
        
        
        
if __name__ == '__main__':
    remove_duplicate_pair()