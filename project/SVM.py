#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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


pipe_clf3 = make_pipeline(StandardScaler, SVC (random_state=0)) 
# pipe_clf3 = make_pipeline(StandardScaler(), SVC) 
param_range = [0.001, 0.1, 1,5, 10.0,50,100]
param_grid = [
              {'svc__C': [0.001,0.1,1,5,10,50,100,500,1000,2000], 'svc__gamma': [0.00001,0.0001,0.001,0.1,1,10,50], 'svc__kernel': ['rbf']}]
gs3 = GridSearchCV(pipe_clf3, param_grid=param_grid, scoring='accuracy', cv=4, refit=True, return_train_score=True)
gs3 = gs3.fit(X_trainn, y_trainn)
print("Accuracy on validation set: {:.3f}".format(100*np.mean(gs3.best_score_),2))
print(gs3.best_params_)
print("Accuracy on validation set :",gs3.best_score_) 
print(gs3.best_params_)
results3 = pd.DataFrame(gs3.cv_results_)
scores3 = np.array(results.mean_test_score).reshape(10,7) 
mglearn.tools.heatmap(scores3, xlabel='svc__gamma',
    xticklabels=param_grid[0]['svc__gamma'],
    ylabel='svc__C', yticklabels=param_grid[0]['svc__C'], cmap="viridis")
plt.show()

