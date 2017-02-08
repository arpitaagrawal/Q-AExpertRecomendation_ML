import numpy as np


def get_L1_dist(list1, list2):
    return np.linalg.norm(np.subtract(np.array(list1),np.array(list2)),1)

def get_L2_dist(list1, list2):
    return np.linalg.norm(np.subtract(np.array(list1),np.array(list2)),2)