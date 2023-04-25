#!/usr/bin/env python
# coding: utf-8

# # Applying bagging to classify examples in the Wine dataset
# 

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import mglearn
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
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
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
from scipy.special import comb
import math
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import _name_estimators
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from itertools import product
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from matplotlib import cm
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from itertools import product
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier


import copy
import warnings
warnings.resetwarnings()
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)
from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')
warnings.filterwarnings('ignore')


# In[5]:


import pandas as pd
train_set = pd.read_csv("/home/aimslab-server/Narges/ML/project/train_features.csv", header=None)
x_train = np.array(train_set)
validation_set = pd.read_csv("/home/aimslab-server/Narges/ML/project/train_data.csv", header=None)
Y_train = np.array(validation_set[list(validation_set.columns[-1:])])
Y_train = np.array(Y_train).reshape((-1))
X_trainval, X_test, y_trainval, y_test = train_test_split (x_train, Y_train, random_state=0) 
X_trainn, X_testt, y_trainn, y_testt = train_test_split (x_train, Y_train, random_state=0) 


# In[6]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#encode the class labels into binary format
# le = LabelEncoder()
# y = le.fit_transform(y)

#split the dataset into 80 percent training and 20 percent test datasets
X_train, X_test, y_train, y_test = train_test_split(x_train, Y_train, 
                                         test_size=0.2, 
                                         random_state=1,
                                         stratify=Y_train)


# In[10]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#For the base classifier, we use decision tree (unpruned)
tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=None,
                              random_state=1)
# clf3 = SVC(gamma=0.1,C=5,kernel='rbf',
#                              random_state=1)
# tree=Pipeline([['sc', MinMaxScaler()],
#                   ['clf', clf3]])
bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=2000, 
                        max_samples=1.0, #draw 100% of the number of samples (with replacement) for each
                        max_features=1.0, #use 100% of the number of features (without replacement)
                        bootstrap=True, #sampling with replacement after each
                        bootstrap_features=False, 
                        n_jobs=1, 
                        random_state=1)


# In[12]:


from sklearn.metrics import accuracy_score

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision Tree train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))

bag = bag.fit(X_train, y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)

bag_train = accuracy_score(y_train, y_train_pred) 
bag_test = accuracy_score(y_test, y_test_pred) 
print('Bagging train/test accuracies %.3f/%.3f'
      % (bag_train, bag_test))


# In[74]:




from sklearn.ensemble import AdaBoostClassifier

tree = DecisionTreeClassifier(criterion='entropy', 
                              max_depth=1,
                              random_state=1)
# scaler=MinMaxScaler()
# scaler.fit(X_train)
# X_train_s=scaler.transform (X_train)
# X_test_s = scaler.transform (X_test)
# tree = SVC(gamma=0.1,C=5,kernel='rbf',
#                              random_state=1,
#                              probability=True,
#                             )

# tree = Pipeline([(MinMaxScaler(),('svc', SVC(kernel='rbf', degree=1, probability=True))])

ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=100, 
                         learning_rate=0.1, #shrinks the contribution of each classifier
                         random_state=1)


# In[39]:


tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)
print('Decision train/test accuracies %.3f/%.3f'
      % (tree_train, tree_test))


# In[ ]:


ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred) 
ada_test = accuracy_score(y_test, y_test_pred) 
print('AdaBoost train/test accuracies %.3f/%.3f'
      % (ada_train, ada_test))





# In[ ]:




