import pandas as pd
import numpy as np
#import pickle
import sys
import os

from algorithms.dilca_dp_final import Dilca_DP
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")



name = sys.argv[1]       
method = sys.argv[2] # 'M','K','K_dep'
n_iter = int(sys.argv[3])
random_seed = 0

#name = 'mushrooms'
#INPUT: file
ext = 'data'
sep = ','
#method = 'K'

#INPUT: model
dic = {'M':'L', 'K':'E', 'K_dep':'E_tensor' }
dic_k = {'dermatology':6,'soybean':19,'mushrooms':2,'cmc':3,'adult':2,'NLTCS':3,'IPUMS_BR':3,'IPUMS_ME':3}
dic_target = {'dermatology':-1,'soybean':0,'mushrooms':0,'cmc':-1,'adult':-1,'NLTCS':99,'IPUMS_BR':99,'IPUMS_ME':99}
k_clust = dic_k[name]

adult = pd.read_table(name+'.'+ext, header = None, sep = sep, skipinitialspace=True)


model = Dilca_DP(dp = False, method = 'M', sigma = 1, k = 3, dp_method = dic[method], eps = .1, h = .3, missing_values = '?', mv_handling='mean_mode', discretize = 'kmeans',n_bins = 5, dtypes = 'keep')    
if dic_target[name] == 0:
    X = adult.iloc[:,1:]
    y = adult.iloc[:,0]
elif dic_target[name] == -1:
    X = adult.iloc[:,:-1]
    y = adult.iloc[:,-1]
else:
    X = adult.iloc[:,:]

model._init_all(X, y = None)
X = model._dataset.copy()

c = True
rs = random_seed -1
for i in range(10):
    if c:
        
        if dic_target[name] < 99: 
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .3, stratify = y, random_state = rs + 1)
        else:
            raise ValueError('no classes available for this dataset')
        
        c = model.check_dataset(X_test, train_dataset = X_train)
        rs += 1

if c:
    raise ValueError('Unable to correctly split the dataset')

model = Dilca_DP(preprocessing = False, dp = False, method = 'M', sigma = 1, k = 3, dp_method = dic[method], eps = .1, h = .3, missing_values = '?', mv_handling='mean_mode', discretize = 'kmeans',n_bins = 5, dtypes = 'keep')    
model.fit(X_train)


new_test = model.encoding(dataset = X_test)
new_train = model.encoding(dataset = X_train)

neigh = knn(5, metric = 'euclidean')

f = open('./output/'+ name + '_'+ method +'_kNN.csv', 'a',1)
f.write(f'method,eps,acc,acc_non_dp\n')

for eps in [.1,.2,.3,.5,.75,1,1.5,2,2.5]:
    for i in range(n_iter):
        t = eps/model._m     
        model.eps = t
        model.fit_dp(model._dataset)     
        new_dp_train = model.encoding(dataset = X_train, distance_list = model._distance_list_dp)
        new_dp_test = model.encoding(dataset = X_test, distance_list = model._distance_list_dp)

        if eps == .1:
            neigh.fit(new_train, y_train)
            labels_knn_ = neigh.predict(new_test)
        a_true = acc(y_test,labels_knn_)
        neigh.fit(new_dp_train, y_train)
        labels_dp_knn_ = neigh.predict(new_dp_test)
        a = acc(y_test, labels_dp_knn_)

            
                   

        f.write(f'{method},{eps},{a}, {a_true}\n')

f.close()

