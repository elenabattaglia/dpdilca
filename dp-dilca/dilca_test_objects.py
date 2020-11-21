import pandas as pd
import numpy as np
#import pickle
import sys
from algorithms.dilca_dp_final import Dilca_DP
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import euclidean_distances
from utils import CreateOutputFile
import warnings
warnings.filterwarnings("ignore")

#INPUT: experiments
name = sys.argv[1]
target = int(sys.argv[2]) #-1 or 0 or 99
n_iter = 30
test = 'objects'

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
h = .3

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
    
del(adult)


def overlap(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    m = min(len(list(set(list1))),len(list(set(list2))))
    
    return float(intersection) / m

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


f, dt = CreateOutputFile(name+'_objects', own_directory = False, date = False, overwrite = False)
dic = {"L":"M", "E":"K", "E_tensor":"K_dep"}

for dp in ["L","E","E_tensor"]:
    #filename = './output/models/'+ name + '_'+dic[dp] + '_clusters_model.sav' 
    #model = pickle.load(open(filename, 'rb'))
    #obj_distance_matrix = model._obj_distance_matrix
    model = Dilca_DP(dp = False, method = dic[dp], sigma = 1, k = 3, dp_method = dp, eps = 1, h = h, missing_values = missing_values, mv_handling=mv_handling, discretize = discretize,n_bins = n_bins)
    model.fit(X)
    new = model.encoding()
    obj_dist_matrix = euclidean_distances(new)
    flatten = obj_dist_matrix[np.triu_indices(model._n)]
    del(obj_dist_matrix, new)

    
    for t in [1,2,3,4,5,7.5,10,15,20,25]:           
        eps = t/(10*model._m)
                
        for s in range(n_iter):
    
            model.fit_dp(model._dataset)
            new_dp = model.encoding(distance_list=model._distance_list_dp)
            obj_dist_matrix_dp = euclidean_distances(new_dp)
            del(new_dp)
            flatten_dp = obj_dist_matrix_dp[np.triu_indices(model._n)]
            del(obj_dist_matrix_dp)
            
            p = pearsonr(flatten,flatten_dp)[0]
            d = np.mean(abs(flatten-flatten_dp))
            f.write(f"{dic[dp]}, {dp}, {t/10}, {sigma}, {h},,,{p},{d},,,,, {dt}\n")


f.close()
