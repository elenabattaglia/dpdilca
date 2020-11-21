import pandas as pd
import numpy as np
#import pickle
import sys
import os

from algorithms.dilca_dp_final import Dilca_DP
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

import warnings
warnings.filterwarnings("ignore")



name = sys.argv[1]       
method = sys.argv[2] # 'M','K','K_dep'
n_iter = int(sys.argv[3])


#INPUT: file
ext = 'data'
sep = ','

#INPUT: model
dic = {'M':'L', 'K':'E', 'K_dep':'E_tensor' }
dic_k = {'dermatology':6,'soybean':19,'mushrooms':2,'cmc':3,'adult':2,'NLTCS':3,'IPUMS_BR':3,'IPUMS_ME':3}
dic_target = {'dermatology':-1,'soybean':0,'mushrooms':0,'cmc':-1,'adult':-1,'NLTCS':99,'IPUMS_BR':99,'IPUMS_ME':99}
k_clust = dic_k[name]

adult = pd.read_table(name+'.'+ext, header = None, sep = sep, skipinitialspace=True)


model = Dilca_DP(dp = False, method = 'M', sigma = 1, k = 3, dp_method = dic[method], eps = .1, h = .3, missing_values = '?', mv_handling='mean_mode', discretize = 'kmeans',n_bins = 5, dtypes = 'keep')    
if target == -1: # the last column contains classes and must be excluded from computations
    X = adult.iloc[:,:-1]
    y = adult.iloc[:,-1]
elif target == 0:# the first column contains classes and must be excluded from computations
    X = adult.iloc[:,1:]
    y = adult.iloc[:,0]
else:            
    raise ValueError('The dataset must have a column containing classes (0: the first column, -1: the last column)') 

model.fit(X)
new = model.encoding()

clustAgg = AgglomerativeClustering(k_clust)
labels_Agg_ = clustAgg.fit_predict(new)
ari_Agg = ari(y, labels_Agg_)
nmi_Agg = nmi(y, labels_Agg_)

f = open('./output/'+ name + '_'+ method +'_clustering.csv', 'a',1)
f.write(f'method,eps,ari_Agg, nmi_Agg\n')
f.write(f'{method},{0},{ari_Agg},{nmi_Agg}\n')

for eps in [.1,.3,.5,.75,1,1.5,2,2.5]:
    for i in range(n_iter):
        t = eps/model._m     
        model.eps = t
        model.fit_dp(model._dataset)     
        new_dp = model.encoding(dataset = [],distance_list = model._distance_list_dp)
        labels_dp_Agg_ = clustAgg.fit_predict(new_dp)

        ari_Agg = ari(y, labels_dp_Agg_)
        nmi_Agg = nmi(y, labels_dp_Agg_)

            
                    

        #f.write(f'{method},{eps},{a},{a_lab},{ari_KM},{ari_lab_KM},{nmi_KM},{nmi_lab_KM},{ari_Agg},{ari_lab_Agg},{nmi_Agg},{nmi_lab_Agg}\n')
        f.write(f'{method},{eps},{ari_Agg},{nmi_Agg}\n')


f.close()


