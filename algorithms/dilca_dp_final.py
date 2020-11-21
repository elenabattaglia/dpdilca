import numpy as np
from algorithms.dilca import Dilca
from scipy.special import binom
from itertools import combinations

class Dilca_DP(Dilca):
    def __init__(self, eps = 1, h = .3, dp_method = 'L', k = 3, method = 'M', sigma = 1, missing_values=None, mv_handling='delete_rows', dtypes='infer', discretize='kmeans', n_bins = 5, dp = False, preprocessing = True):
        Dilca.__init__(self, method, sigma, k, missing_values, mv_handling, dtypes, discretize, n_bins, dp, preprocessing)
        self.eps = eps
        self.h = h
        self.dp_method = dp_method
        


    def fit_dp(self, X):
        if not self._fitted:
            self.fit(X)
        self._context_dp = []
        self._gs = 1/self._n*(1/np.log(2) + np.log2(self._n))
        if self.dp_method == 'L':
            
            self._entropies_dp = np.zeros(self._m)
            self._joint_entr_matrix_dp = np.zeros((self._m,self._m))
            self._su_matrix_dp = np.zeros((self._m,self._m))
            for i in range(self._m):
                self._entropies_dp[i] = self._entropies[i] + np.random.laplace(0,self._gs*(2*self._m-1)/(self.eps*self.h))
                for j in range(i+1, self._m):
                    self._joint_entr_matrix_dp[i,j] = self._joint_entr_matrix_dp[j,i] = self._joint_entr_matrix[i,j] + np.random.laplace(0,self._gs*(2*self._m-1)/(self.eps*self.h))

            for i in range(self._m):
                for j in range(i+1, self._m):
                    self._su_matrix_dp[i,j] = self._su_matrix_dp[j,i] = 1 - self._joint_entr_matrix_dp[i,j]/(self._entropies[i] + self._entropies[j])
                    
            for i in range(self._m):
                self._context_dp.append(self._context_M(i, self.sigma, self._su_matrix_dp))
            


        elif self.dp_method == 'E':

            for i in range(self._m):
                self._context_dp.append(self._context_E(i, self.k))

        elif self.dp_method == 'E_tensor':
            for i in range(self._m):
                self._context_dp.append(self._context_E_tensor(i, self.k))
             
        

        self._distance_list_dp = []
        for i in range(self._m):
            ct_dp = [np.copy(a) for a in self._contingency_tables[i]]
            #ct_dp = [np.copy(a) for a in self._contingency_tables[i]]      #forse ora non serve più copiare le contingency tables
            eps_cm = self.eps*(1-self.h)/len(self._context_dp[i])
            for c in self._context_dp[i]:
                if c > i:
                    L = np.random.laplace(0,2/eps_cm, ct_dp[c].shape)
                    ct_dp[c] = np.add(ct_dp[c],L,casting='safe')
                if c < i:
                    #print(ct_dp[c])
                    ct_dp[c] = np.copy(self._contingency_tables[c][i].transpose())
                    #print(ct_dp[c])
                    L = np.random.laplace(0,2/eps_cm, ct_dp[c].shape)
                    ct_dp[c] = np.add(ct_dp[c],L,casting='safe')
                ct_dp[c] = ct_dp[c] * (ct_dp[c] > 0)
                #print(ct_dp[c].shape)
                
            self._distance_list_dp.append(self._distance(i, T_list = ct_dp, c = self._context_dp))        


    def _context_E(self, target, k):
        su = self._mi_matrix[:,target]
        mi_max = np.max([s for j,s in enumerate(su) if j!=target])
        t = 300 - self.eps*self.h*(mi_max)/(4*self._gs*k)
                
        p = np.exp(self.eps*self.h*(su)/(4*self._gs*k) + t)
        p[target] = 0
        s = np.sum(p)
        p = p/s
        #self._p = p
        #context = []
        if len(p[p>0]) < k:
            k = len(p[p>0])
        context= np.random.choice(np.arange(self._m), size = k, replace=False, p = p)
        context.sort()
        return list(context)

    def _context_E_tensor(self, target, k):
        r = [j for j in range(self._m) if j != target]
        mi = np.zeros(int(binom(self._m - 1, k)), dtype = np.float64)
        s = []
        mi_max = -np.inf
        mi_min = 0
        for j,i in enumerate(combinations(r,k)):
            T = self._contingency_tensor([target] + [j for j in i])
            T1 = np.sum(T, axis = 0)
            out = np.zeros(T.shape)
            mi[j] = np.sum(T/self._n*np.log2(T/T1, out = out, where=T>0))
            #print(j, mi)
            #p[j] = self.eps*self.h*(mi)/(4*self._gs)
            
            s.append(i)
            if mi[j] > mi_max :
              mi_max = mi[j]
            if mi[j] < mi_min :
              mi_min = mi[j]
        #print(mi_max - mi_min, mi_min, mi_max)
        t = 300 - self.eps*self.h*(mi_max)/(4*self._gs)
        #self._p = np.exp(self.eps*self.h*(mi)/(4*self._gs) + t)
        p = np.exp(self.eps*self.h*(mi)/(4*self._gs) + t)
        
        context= np.random.choice(np.arange(len(s)), p = p/np.sum(p))
        return list(s[context])
            
            
