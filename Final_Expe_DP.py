import pandas as pd
import numpy as np
import pickle
import sys

from algorithms.dilca_dp_final import Dilca_DP
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
import diffprivlib as dp

import warnings
warnings.filterwarnings("ignore")


#INPUT: file
name = sys.argv[1]
method = sys.argv[2]
n_iter = int(sys.argv[3])

ext = 'data'
sep = ','

#INPUT: model
#method = 'K' #'MI_3','K_dep_3'
dic = {'M':'L', 'K':'E', 'K_dep':'E_tensor' }
dic_k = {'dermatology':6,'soybean':19,'mushrooms':2,'cmc':3,'adult':2,'NLTCS':3,'IPUMS_BR':3,'IPUMS_ME':3}
dic_target = {'dermatology':-1,'soybean':0,'mushrooms':0,'cmc':-1,'adult':-1,'NLTCS':99,'IPUMS_BR':99,'IPUMS_ME':99}


adult = pd.read_table(name+'.'+ext, header = None, sep = sep, skipinitialspace=True)
#filename = './output/models/'+ name + '_'+ method + '_fitted_model.sav'

if dic_target[name] == 0:
    X = adult.iloc[:,1:]
    y = adult.iloc[:,0]
elif dic_target[name] == -1:
    X = adult.iloc[:,:-1]
    y = adult.iloc[:,-1]
else:
    X = adult.iloc[:,:]
k_clust = dic_k[name]

model = Dilca_DP(dp = False, method = method, sigma = 1, k = 3, dp_method = dic[method], eps = .1, h = .3, missing_values = '?', mv_handling='mean_mode', discretize = 'kmeans',n_bins = 5, dtypes = 'infer')    

model.fit(X)      
new = model.encoding()


# OneHot encoding
X_list = []
for k in range(model._m):
    _k = len(set(model._dataset[k]))
    #a = np.zeros((1,model._k[k] -1))
    b = np.diag([1]*(_k))
    #X = np.concatenate((a,b))
    X_list.append(b)
new_oh = pd.DataFrame()
for f in range(model._m):
    for k in range(X_list[f].shape[1]):
        new_oh[f'{f}_{k}'] = model._dataset.apply(lambda x: X_list[f][x.iloc[f],k], axis = 1)

f = open('./output/'+ name + '_'+ method +'_DP_Kmeans.csv', 'a',1)
f.write(f'method,eps,ari_dp,nmi_dp,ari,nmi,ari_oh, nmi_oh, NICV_dp,NCIV, NICV_oh\n')

t = 2
for eps in [.1, .3, .5, .75, 1, 1.5, 2, 2.5]:
    a,n,a_dp,n_dp,a_oh_dp,n_oh_dp, c_dp, c_oh_dp, c = (0,0,0,0,0,0,0,0,0)
    m_ = np.array(new.apply(lambda x:min(x), axis = 0))
    M_ = np.array(new.apply(lambda x:max(x), axis = 0))

    for i in range(n_iter):
        model.eps = eps
        model.fit_dp(model._dataset)
        new_dp = model.encoding(dataset = [], distance_list = model._distance_list_dp)

        m = np.array(new_dp.apply(lambda x:min(x), axis = 0))
        M = np.array(new_dp.apply(lambda x:max(x), axis = 0))

        clust_dp = dp.models.KMeans(epsilon=t, n_clusters = k_clust, bounds=(m,M), random_state = 0)
        clust_dp_oh = dp.models.KMeans(epsilon=eps+t, n_clusters = k_clust, bounds = (np.zeros(new_oh.shape[1]),np.ones(new_oh.shape[1])), random_state = 0)
        clust_non_dp = dp.models.KMeans(epsilon=t, n_clusters = k_clust, bounds = (m_,M_), random_state = 0)
                
        lab_non_dp = clust_non_dp.fit_predict(new)
        a += ari(y,lab_non_dp)
        n += nmi(y,lab_non_dp)
        lab = clust_dp.fit_predict(new_dp)
        a_dp += ari(y,lab)
        n_dp += nmi(y,lab)
        lab_oh = clust_dp_oh.fit_predict(new_oh)
        a_oh_dp += ari(y,lab_oh)
        n_oh_dp += nmi(y,lab_oh)
        c_dp += clust_dp.inertia_/(np.sum((M-m)**2)*model._n)
        c_oh_dp +=clust_dp_oh.inertia_/((new_oh.shape[1])*model._n)
        c += clust_non_dp.inertia_/(np.sum((M_-m_)**2)*model._n)

    f.write(f'{method},{eps},{a_dp/n_iter},{n_dp/n_iter},{a/n_iter},{n/n_iter},{a_oh_dp/n_iter},{n_oh_dp/n_iter},{c_dp/n_iter},{c/n_iter},{c_oh_dp/n_iter}\n')
        

f.close()



