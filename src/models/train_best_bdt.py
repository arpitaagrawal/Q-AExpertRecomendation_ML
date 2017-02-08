'''
Trains a boosting tree
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

from models.train_bdt import run_BDT

run_BDT(RandomForestClassifier, 2, 50, 1, save=True, test=False)
            
            