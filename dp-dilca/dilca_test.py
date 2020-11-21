import pandas as pd
import numpy as np
#import pickle
import sys

from algorithms.dilca_dp_final import Dilca_DP
from scipy.stats import pearsonr
from utils import CreateOutputFile
import warnings
warnings.filterwarnings("ignore")

#INPUT: experiments
n_iter = 30
name = sys.argv[1]
target = int(sys.argv[2]) #-1 or 0 or 99
test = sys.argv[3] #'result' or 'cont'

#INPUT: file
ext = 'data'
sep = ','
missing_values = '?'
mv_handling = 'mean_mode'
discretize = 'kmeans'
n_bins = 5

#INPUT: algorithm
sigma = 1
k = 3


adult = pd.read_table(name+'.'+ext, header = None, sep = sep, skipinitialspace=True)
if target == -1: # the last column contains classes and must be excluded from computations
    print(-1)
    X = adult.iloc[:,:-1]
elif target == 0:# the first column contains classes and must be excluded from computations
    print(0)
    X = adult.iloc[:,1:]
else:            # all columns must be included
    print(99)
    X = adult
    

def overlap(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    m = min(len(list(set(list1))),len(list(set(list2))))
    
    return float(intersection) / m

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

if test == 'cont':
    f, dt = CreateOutputFile(name+'_cont', own_directory = False, date = False, overwrite = False)
    h = .5
else:
    f, dt = CreateOutputFile(name, own_directory = False, date = False, overwrite = False)
    h = .3

for t in [1,2,3,4,5,7.5,10,15,20,25]:
    for dp in ["L","E","E_tensor"]:
        dic = {"L":"M", "E":"K", "E_tensor":"K_dep"}
        if test == 'cont':
            eps = t/5
        else:
            eps = t/10
        model = Dilca_DP(dp = False, method = dic[dp], sigma = 1, k = k, dp_method = dp, eps = eps, h = h, missing_values = missing_values, mv_handling=mv_handling, discretize = discretize,n_bins = n_bins)
        #filename = './output/models/'+ name + '_'+dic[dp] + '_fitted_model.sav' 
        #model_ = pickle.load(open(filename, 'rb'))
        model.fit(X)
        
        for s in range(n_iter):
    
            model.fit_dp(X)
            
            for i in range(model._m):
                if i == 0:
                    k = model._contingency_tables[i][i+1].shape[0]
                else:
                    k = model._contingency_tables[i-1][i].shape[1]

                p = pearsonr(model._distance_list[i].flatten(),model._distance_list_dp[i].flatten())[0]
                d = np.mean(abs(model._distance_list[i]-model._distance_list_dp[i]))
                j = jaccard_similarity(model._context[i],model._context_dp[i])
                o = overlap(model._context[i],model._context_dp[i])
                c = '-'.join(str(n) for n in model._context[i])
                c_dp = '-'.join(str(n) for n in model._context_dp[i])

                f.write(f"{dic[dp]}, {dp}, {t/10}, {sigma}, {h},{i}, {k},{p},{d},{c}, {c_dp}, {j}, {o}, {dt}\n")


f.close()
