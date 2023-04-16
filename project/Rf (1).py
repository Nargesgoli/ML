#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# In[ ]:


train_set = pd.read_csv("/home/aimslab-server/Narges/ML/project/train_features.csv", header=None)
x_train = np.array(train_set)
validation_set = pd.read_csv("/home/aimslab-server/Narges/ML/project/train_data.csv", header=None)
Y_train = np.array(validation_set[list(validation_set.columns[-1:])])
Y_train = np.array(Y_train).reshape((-1))


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


## pca (n_components=200)
pca1 = PCA(n_components=200)
X_train_pca = pca1.fit_transform(X_trainval)

print("pca.components_.shape: {}".format(pca1.components_.shape))


# In[ ]:


## data mapping
kmeans = KMeans(n_clusters=200, random_state=0)
kmeans.fit(X_trainval)
# y_pred_map = kmeans.predict(X_trainval)
distance_features = kmeans.transform(X_trainval)
print("Actual feature shape: {}".format(X_trainval.shape))
print("Distance feature shape: {}".format(distance_features.shape))


# In[ ]:


param_grid = {'n_estimators': [1, 10, 100,200,500],
              'max_depth':[1,2,3,4,5,6,10,15,20,50,100,1000],
              'max_features':[1,2,3,4,5,6,7,8]} 
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=4)
grid_search.fit(X_train, y_train)
print("Test set score: {:.2f}".format(grid_search.score(X_valid, y_valid)))


# In[ ]:




