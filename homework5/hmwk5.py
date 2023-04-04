#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


X_train = pd.read_csv("/Users/nrgsg/Desktop/courses/CSE-546/Hwk/homework5/dataset_hwk5/X.csv", header=None)
X = np.array(X_train)
imagee = pd.read_csv("/Users/nrgsg/Desktop/courses/CSE-546/Hwk/homework5/dataset_hwk5/images.csv", header=None)
images=np.array(imagee)


# In[3]:


print("X_train.shape=",X.shape)


# In[97]:


plt.figure(figsize=(50, 5))
plt.boxplot(X)
# plt.xticks(range(1,999),list(X_testt.columns[1:-1]), rotation=90)
plt.show()


# In[4]:


# #Normalization
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

scaler=Normalizer()
scaler.fit(X)
X = scaler.transform (X)
# xt= scaler.transform (Xt)


# In[5]:


# plt.imshow(images[0].reshape(32,32,3))


# In[6]:


distortions = [ ]
for i in range(2, 21):
#     km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km = KMeans( n_clusters=i,random_state=0)
    km.fit(X)
    distortions.append(km.inertia_) 
plt.plot(range(2,21), distortions, marker='o') 
plt.xlabel('Number of clusters') 
plt.ylabel('Distortion')
plt.tight_layout()
plt.show()


# In[7]:


# km = KMeans( init='k-means++', n_init=10, max_iter=300, random_state=0)
# params = {'n_clusters': [2,3,4,5,6,7,8,9,10] }

# grid_kmt = GridSearchCV(param_grid=params, estimator=km,scoring = silhouette)
# grid_kmt.fit(X)


# In[122]:


km1 = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# km1 = KMeans(n_clusters=5)
y_km = km1.fit_predict(X)
cluster_labels = np.unique(y_km)
# print("cluster_labels=",cluster_labels.shape[0])
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km,metric='euclidean')
# print("silhouette_vals[y_km == c]",(silhouette_vals[y_km == 0]))
print("silhouette_vals=",silhouette_vals.shape)
y_ax_lower, y_ax_upper = 0, 0 
yticks = []
for j, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals_org=np.copy (c_silhouette_vals)
    c_silhouette_vals.sort()
    c_silhouette_vals_max=c_silhouette_vals[-6:-1]
    print("cluster=",j)
    #print("silhouette values max=",c_silhouette_vals_max.index)
    index_max=[]
    for k in c_silhouette_vals_max:
        idx=(list(silhouette_vals)).index(k)
        index_max.append(idx)
#         print("idx_max=",idx)
    print("index_max=",index_max)
    c_silhouette_vals_minn=[]
    index_min=[]
    for m in c_silhouette_vals:
        if m > 0 :
               c_silhouette_vals_minn.append(m)
    c_silhouette_vals_min=c_silhouette_vals_minn[:2]
    for h in c_silhouette_vals_min:
        idx=(list(silhouette_vals)).index(h)
        index_min.append(idx)
    print("index_min=",index_min)
#     print("c_silhouette_vals_min=",c_silhouette_vals_minn[:3])
    
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(j) / n_clusters) 
    plt.barh (range(y_ax_lower,y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none', color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2.)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals) 
plt.axvline (silhouette_avg, color="red",linestyle="--") 
plt.yticks(yticks, cluster_labels ) 
plt.ylabel('Cluster') 
plt.xlabel('Silhouette coefficient') 
plt.tight_layout()
plt.show()


# In[86]:


plt.imshow(images[1516].reshape(32,32,3))
plt.imshow(images[4286].reshape(32,32,3))


# In[111]:


silhouette_avg=[]
ns=range(10,100,2)
for n in ns:
    km_n = KMeans(n_clusters=5, init='k-means++', n_init=n, max_iter=300, tol=1e-04, random_state=0)
    # km1 = KMeans(n_clusters=5)
    y_km_n = km_n.fit_predict(X)
    cluster_labels = np.unique(y_km)
# print("cluster_labels=",cluster_labels.shape[0])
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km_n,metric='euclidean')
    silhouette_avg .append(np.mean(silhouette_vals))
plt.plot(ns, silhouette_avg, marker='o') 
plt.xlabel('n_init') 
plt.ylabel('silhouette_avg')
plt.tight_layout()
plt.show()    


# In[121]:


silhouette_avg=[]
ms=[300,400,500,1000,2000,3000,5000,6000,10000]
j=len(ms)
for n in ms:
    km_n = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=n, tol=1e-04, random_state=0)
    # km1 = KMeans(n_clusters=5)
    y_km_n = km_n.fit_predict(X)
    cluster_labels = np.unique(y_km)
# print("cluster_labels=",cluster_labels.shape[0])
    n_clusters = cluster_labels.shape[0]
    silhouette_vals = silhouette_samples(X, y_km_n,metric='euclidean')
    silhouette_avg .append(np.mean(silhouette_vals))
plt.plot(ms, silhouette_avg, marker='o')
plt.xlabel('n_init') 
plt.ylabel('silhouette_avg')
plt.tight_layout()
plt.show() 


# # agglomerative 

# In[148]:


from scipy.cluster.hierarchy import dendrogram, ward
linkage_array = ward (X)
dendrogram (linkage_array)
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [10.75, 10.75], '--', c='k')
# ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 10.75, 'five clusters',va='center') 
# ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")


# In[165]:


from scipy.cluster.hierarchy import dendrogram, ward, single,linkage
linkage_array1 = linkage(X,'complete')
print(len(linkage_array1))
# plt.figure(figsize=(20,15))
dendrogram (linkage_array1)
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [1.3, 1.3], '--', c='k')
# # ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 1.3, 'five clusters',va='center') 
# # ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
# plt.xlabel("Sample index")
plt.ylabel("Cluster distance")


# In[6]:


from scipy.cluster.hierarchy import dendrogram, ward, single,linkage
linkage_array2 = linkage(X,'single')
print(len(linkage_array2))
plt.figure(figsize=(15, 30))
dendrogram (linkage_array2)
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [1.3, 1.3], '--', c='k')
# # ax.plot(bounds, [4, 4], '--', c='k')
ax.text(bounds[1], 1.3, 'five clusters',va='center') 
# # ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
# plt.xlabel("Sample index")
plt.ylabel("Cluster distance")


# In[175]:


from sklearn.cluster import AgglomerativeClustering 
agg = AgglomerativeClustering (n_clusters=5,linkage='ward') 
y_agg = agg.fit_predict(X)
# print("y_agg=",y_agg)
cluster_labels_agg = np.unique(y_agg)
# print("cluster_labels=",cluster_labels.shape[0])
n_clusters = cluster_labels_agg.shape[0]
silhouette_vals_agg = silhouette_samples(X, y_agg,metric='euclidean')
# print("silhouette_vals[y_km == c]",(silhouette_vals[y_km == 0]))
print("silhouette_vals=",silhouette_vals_agg.shape)
y_ax_lower, y_ax_upper = 0, 0 
yticks = []
for j, c in enumerate(cluster_labels_agg):
    c_silhouette_vals_agg = silhouette_vals_agg[y_agg == c]
    c_silhouette_vals_agg_org=np.copy (c_silhouette_vals_agg)
    c_silhouette_vals_agg.sort()
    c_silhouette_vals_max=c_silhouette_vals_agg[-6:-1]
    print("cluster=",j)
    #print("silhouette values max=",c_silhouette_vals_max.index)
    index_max=[]
    for k in c_silhouette_vals_max:
        idx=(list(silhouette_vals_agg)).index(k)
        index_max.append(idx)
#         print("idx_max=",idx)
    print("index_max=",index_max)
    c_silhouette_vals_minn=[]
    index_min=[]
    for m in c_silhouette_vals_agg:
        if m > 0 :
               c_silhouette_vals_minn.append(m)
    c_silhouette_vals_min=c_silhouette_vals_minn[:2]
    for h in c_silhouette_vals_min:
        idx=(list(silhouette_vals_agg)).index(h)
        index_min.append(idx)
    print("index_min=",index_min)
#     print("c_silhouette_vals_min=",c_silhouette_vals_minn[:3])
    
    y_ax_upper += len(c_silhouette_vals_agg)
    color = cm.jet(float(j) / n_clusters) 
    plt.barh (range(y_ax_lower,y_ax_upper), c_silhouette_vals_agg, height=1.0, edgecolor='none', color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2.)
    y_ax_lower += len(c_silhouette_vals_agg)
silhouette_avg_agg = np.mean(silhouette_vals_agg) 
plt.axvline (silhouette_avg_agg, color="red",linestyle="--") 
plt.yticks(yticks, cluster_labels_agg ) 
plt.ylabel('Cluster') 
plt.xlabel('Silhouette coefficient') 
plt.tight_layout()
plt.show()


# In[ ]:




