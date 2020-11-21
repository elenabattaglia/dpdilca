from algorithms.CreateMatrix import CreateMatrix
import numpy as np
from algorithms.dilca_dp_final import Dilca_DP
import warnings
warnings.filterwarnings("ignore")


shape = 'A'
sigma = 1
h = .5
dp = 'L'

missing_values = '?'
mv_handling = 'mean_mode'
discretize = 'kmeans'
n_bins = 5

for j, n in enumerate([0.1,0.2,.3]): 

    if shape == 'A':
        V,x,y = CreateMatrix(v_rowclust = [5000,5000], v_colclust=[2,2],noise = n)
        T = np.random.choice([0,1],size = (10000,7))
        X = np.concatenate((V,T), axis = 1)
        s = set([1,2,3])
        k = 3
    
    if shape == 'B':
        V,x,y = CreateMatrix(v_rowclust = [8000,2000], v_colclust=[2,2],noise = n)
        T = np.random.choice([0,1],size = (10000,7))
        X = np.concatenate((V,T), axis = 1)
        s = set([1,2,3])
        k = 3

    if shape == 'C':
        V,x,y = CreateMatrix(v_rowclust = [7000,3000], v_colclust=[2,2],noise = n)
        U, x, y = CreateMatrix(v_rowclust = [7000,3000], v_colclust=[1,1],noise = .35)
        T = np.random.choice([0,1],size = (10000,5))
        X = np.concatenate((V,U,T), axis = 1)
        s = set([1,2,3])
        k = 3

    if shape == 'D':
        V,x,y = CreateMatrix(v_rowclust = [2000,2000,2000,2000,2000], v_colclust=[1,1,1,1,1],noise = n)
        T = np.random.choice([0,1],size = (10000,6))
        X = np.concatenate((V,T), axis = 1)
        s = set([1,2,3,4])
        k = 4


    
    f = open('./output/sinth' + shape+'.csv', 'a',1)
    eps_ = [.1,.2,.3,.4,.5,.75,1.0,1.5,2.0,2.5]
    for t in eps_:

        eps = t*2
        model = Dilca_DP(method = "M", dp = True, sigma = 1, k = k, dp_method = "L", eps = eps, h = h, missing_values = missing_values, mv_handling=mv_handling, discretize = discretize,n_bins = n_bins, dtypes = dtypes)
        
                    
        cE_tensor = 0
        cE = 0
        cL = 0
        #cL1 = 0
        #c1 = 0
        for i in range(100):
            model.fit_dp(X)
            contextE = model._context_E(0,k)
            contextE_tensor = model._context_E_tensor(0,k)

            if set(model._context_dp[0]) == s:
                cL += 1
            #if set([1,2]).issubset(model._context_dp[0]):
            #    cL1 += 1

            if set(contextE_tensor)==s:
                cE_tensor +=1
            if set(contextE)==s:
                cE+=1
        f.write(f'L,{n},{t},{cL}\n')
        f.write(f'E,{n},{t},{cE}\n')
        f.write(f'E_tensor,{n},{t},{cE_tensor}\n')


f.close()    

