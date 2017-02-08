import numpy as np


user_train_data_set = []
ques_train_data_set = []

def load_user_data_set():
    global user_train_data_set
    myFile = open("../../bytecup2016data/user_info.txt", "r")
    myString = myFile.read()
    myList = myString.split("\n")
    
    user_train_data_set = np.empty([len(myList), 4], dtype=np.dtype((str, 150)))
    for row in range(0,len(myList)):
        arr = myList[row].split("\t")
        user_train_data_set[row] = arr
    
def question_data_set():
    global ques_train_data_set
    myFile = open("../../bytecup2016data/question_info.txt", "r")
    myString = myFile.read()
    myList = myString.split("\n")
    
    ques_train_data_set = np.empty([len(myList), 7], dtype=np.dtype((str, 150)))
    for row in range(0,len(myList)):
        arr = myList[row].split("\t")
        ques_train_data_set[row] = arr

load_user_data_set()
question_data_set()

#print user_train_data_set[0]
#print np.unique(ques_train_data_set[:,1])