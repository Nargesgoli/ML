#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering 
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import copy
import warnings
warnings.resetwarnings()
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)
from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')
warnings.filterwarnings('ignore')


# In[ ]:


train_set = pd.read_csv("/home/aimslab-server/Narges/ML/project/train_features.csv", header=None)
x_train = np.array(train_set)
validation_set = pd.read_csv("/home/aimslab-server/Narges/ML/project/train_data.csv", header=None)
Y_train = np.array(validation_set[list(validation_set.columns[-1:])])
Y_train = np.array(Y_train).reshape((-1))
X_trainval, X_test, y_trainval, y_test = train_test_split (x_train, Y_train, random_state=0)
X_trainn, X_testt, y_trainn, y_testt = train_test_split (x_train, Y_train, random_state=0) 


# In[ ]:


from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler.fit(X_trainval)
# X_train = scaler.transform (X_train)
X_trainval = scaler.transform (X_trainval)
X_test = scaler.transform (X_test)
X_train, X_valid, y_train, y_valid = train_test_split (X_trainval, y_trainval, random_state=1)
print("X_train.shape:",X_train.shape)
print("X_valid.shape:",X_valid.shape)


# In[ ]:


pipe_clf3 = make_pipeline(MinMaxScaler(), RandomForestClassifier()) 
# n_est = [2,3,4,5,6,10,15,20,30,50,60,70,80,90,100,1000]
param_grid = [{'randomforestclassifier__n_estimators':[3,5,10,12,15,20,25,50,150,100,200,500],
               'randomforestclassifier__max_depth':[3,5,6,10,12,15,20,50,100],
               'randomforestclassifier__max_features':[4,5,6,7,8,10,12,15,20,50,100],
               'randomforestclassifier__max_leaf_nodes':[2,3,4,5,6,10,15,20,30,50,60,70,80,90,100]}]
gs = GridSearchCV(pipe_clf3, param_grid=param_grid, scoring='accuracy', cv=4, refit=True)
gs = gs.fit(X_trainn, y_trainn)
print("best score:", gs.best_score_) 
print(gs.best_params_)




