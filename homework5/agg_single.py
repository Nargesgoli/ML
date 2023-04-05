#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
import copy
import warnings
warnings.resetwarnings()
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)
from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')


# In[ ]:


X_train = pd.read_csv("/Narges/ML/homework5/X.csv", header=None)
X = np.array(X_train)
print("X_train.shape=",X.shape)


# In[ ]:
import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(100000)

from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler=Normalizer()
scaler.fit(X)
X = scaler.transform (X)


# In[ ]:


from scipy.cluster.hierarchy import dendrogram, ward, single,linkage
linkage_array1 = linkage(X,'single')
print(len(linkage_array1))
# plt.figure(figsize=(20,15))
dendrogram (linkage_array1)
ax = plt.gca()
# bounds = ax.get_xbound()
# ax.plot(bounds, [1.3, 1.3], '--', c='k')
# # ax.plot(bounds, [4, 4], '--', c='k')
# ax.text(bounds[1], 1.3, 'five clusters',va='center') 
# # ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
# plt.xlabel("Sample index")
plt.ylabel("Cluster distance")
plt.show()

