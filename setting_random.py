__author__ = 'westbrick'

import numpy as np
import scipy as sp
import numpy.random as r
import distance
import math



n = 10000
# ds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
# ds = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# ds = [12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
#ds = [4, 8, 16, 32, 4, 8, 16, 32]
ds = [8]
#ds = [4, 8, 16, 32]
#ds = [32]
#ms = [1]
#ds = [2, 4, 8, 16]
#ds = [5]
#ds = [1024-16, 8192-16]
values = [
    [d-i-1 for i in range(d)]
    #[1.0/(i+1) for i in range(d)]
    for d in ds
    #[ds[0]-i-1 for i in range(ds[0])],
    #[1.0/(i+1) for i in range(ds[1])]
]


#eps = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
#eps = [0.01, 0.1, 1.0, 2.0, 3.0]
#eps = [ 0.1, 0.5, 1.0, 2.0, 3.0]
#eps = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
eps = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
#eps = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
#eps = [1.0, 1.5, 2.0, 2.5, 3.0]
# eps = [0.01, 0.1, 0.4, 1.0, 2.0]
#eps = [1.0]
#eps = [0.1, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0, 2.3, 2.6, 3.0]

#na = [int(n*0.0), int(n*0.001), int(n*0.01), int(n*0.1)] # factions of data (a)mplification attacks
#na = [int(n*0.0), int(n*0.5)]
#na = [int(n*0.0), int(n*0.001), int(n*0.01), int(n*0.05)]
na = [1, ]
#nd = [int(n*0.0), int(n*0.001), int(n*0.01), int(n*0.1)] # fractions of view (d)isguise attacks
#nd = [int(n*0.0), int(n*0.05)]
#nd = [int(n*0.0), ]
nd = [int(n*0.0), int(n*0.001), int(n*0.01), int(n*0.05)]
#nd = [int(n*0.01), int(n*0.05)]
repeat = 500
#repeat = 1


permutations = None


#mechanisms = ['MRR', 'BRR', 'GBFMM', 'HBFMM', 'KSS', 'EM', 'EKSE', 'OKSE', 'ECSE', 'OCSE']
#mechanisms = ["LAPLACE", "SAMPLEX0BRR", "SAMPLEX0SUBSET", "SAMPLXLAPLACE", "SAMPLEXBRR", "SAMPLEXSUBSET", "ADDITIVE"]
#mechanisms = ["LAPLACE", "SAMPLEXLAPLACE", "SAMPLEXBRR", "SAMPLEXSUBSET", "ADDITIVE"]
mechanisms = ["LAPLACE", "SAMPLEX0BRR", "SAMPLEXBRR", "SAMPLEX0SUBSET", "SAMPLEXSUBSET", "SAMPLEX1PIECEWISE", "ADDITIVE"]

print('n:',n)
print('ds:', ds)
print('values:', values)
print('eps:', eps)
print('na:', na)
print('nd:', nd)
print('ms:', mechanisms)
