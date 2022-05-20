__author__ = 'westbrick'
# util functions

import math
import numpy as np
import scipy as sp
import numpy.random as r
import laplace
import samplex
import additive
import utils
import decimal
from decimal import Decimal as D
import time
from scipy.stats import rankdata

def binarysearch(l, v):
    # search the corresponding scope holding v
    s = 0
    e = len(l)-1
    while v < l[math.floor((s+e)/2)] or v >= l[math.floor((s+e)/2) + 1]:
        if v < l[math.floor((s+e)/2)]:
            e = math.floor((s+e)/2)
        else:
            s = math.floor((s+e)/2)
    return math.floor((s+e)/2)

def bitarrayToList(ba):
    return [i for i in range(0, len(ba)) if ba[i] > 0]

def reservoirsample(l, m):
    # sample m elements from list l
    samples = l[0:m]
    for i in range(m, len(l)):
        index = r.randint(0, i+1)
        if index < m:
            samples[index] = l[i]
    return samples


def recorder(sums, pub):
    # record pub to hits
    sums += pub
    return sums

def distributor(n, profiles, mechanism):
    # randomize items in the hitogram and return observed hits
    sums = np.full(len(profiles[0]), 0, dtype=float)
    for i in range(0, n):
        pub = mechanism.randomizer(profiles[i])
        tstart = time.process_time()
        recorder(sums, pub)
        mechanism.recordtime += time.process_time()-tstart
        #print("user ", i)
    #print(hits, mechanism.name)
    return mechanism.decoder(sums, n)

def rawDistributor(n, profiles, mechanism):
    # randomize items in the hitogram and return observed hits
    pubs = np.full((n,len(profiles[0])), 0, dtype=float)
    for i in range(0, n):
        pubs[i] = mechanism.randomizer(profiles[i])
        tstart = time.process_time()
        #recorder(sums, pub)
        mechanism.recordtime += time.process_time()-tstart
        #print("user ", i)
    #print(hits, mechanism.name)
    return pubs


def projector(od):
    # project od to probability simplex
    u = -np.sort(-od)
    #print("sorted:\t", u)
    sod = np.zeros(len(od))
    sod[0] = u[0]
    for i in range(1, len(od)):
        sod[i] = sod[i-1]+u[i]

    for i in range(0, len(od)):
        sod[i] = u[i]+(1.0-sod[i])/(i+1)

    p = 0
    for i in range(len(od)-1, -1, -1):
        if sod[i] > 0.0:
            p = i
            break

    q = sod[p]-u[p]

    x = np.zeros(len(od))
    for i in range(0, len(od)):
        x[i] = np.max([od[i]+q, 0.0])
    #print("projected:\t",x)
    return x




def randomPermutations(n, d, value, scale=None, repeat=1):
    permutations = np.zeros((n, d), dtype=int)
    if scale == None:
        scale = r.uniform(0.0, 1.0, d)
    randoms = r.uniform(0.0, 1.0, (n,d))
    for i in range(0, math.ceil(n/repeat)):
        #permutations[i] = r.permutation(d)
        #print(randoms[i], scale, randoms[i]*scale)
        arank = np.array(rankdata(randoms[i]*scale, method='ordinal')-1).astype(int)
        if i == math.ceil(n/repeat):
            permutations[i*repeat:n] = np.array([arank]*(n-i*repeat))
        else:
            permutations[i*repeat:(i+1)*repeat] = np.array([arank]*repeat)
    #print("scale ", scale)
    #print(permutations)
    return permutations






def initmechanisms(mechanisms, d, value, ep):
    # instance mechanisms with concrete setting
    mechanism_instances = []
    for mk in mechanisms:
        if mk == 'LAPLACE':
            mechanism_instances.append(laplace.LAPLACE(d, value, ep))
        elif mk == 'SAMPLEX0BRR':
            mechanism_instances.append(samplex.SAMPLEX(d, value, ep, 0.0, None, "BRR"))
            mechanism_instances[-1].name = mk
        elif mk == 'SAMPLEX0LAPLACE':
            mechanism_instances.append(samplex.SAMPLEX(d, value, ep, 0.0, None, "LAPLACE"))
            mechanism_instances[-1].name = mk
        elif mk == 'SAMPLEX0SUBSET':
            mechanism_instances.append(samplex.SAMPLEX(d, value, ep, 0.0, None, "SUBSET"))
            mechanism_instances[-1].name = mk
        elif mk == 'SAMPLEX1PIECEWISE':
            mechanism_instances.append(samplex.SAMPLEX(d, value, ep, 0.0, [1/d]*d, "PIECEWISE"))
            mechanism_instances[-1].name = mk
        elif mk == 'SAMPLEXBRR':
            mechanism_instances.append(samplex.SAMPLEX(d, value, ep, None, None, "BRR"))
            mechanism_instances[-1].name = mk
        elif mk == 'SAMPLEXLAPLACE':
            mechanism_instances.append(samplex.SAMPLEX(d, value, ep, None, None, "LAPLACE"))
            mechanism_instances[-1].name = mk
        elif mk == 'SAMPLEXSUBSET':
            mechanism_instances.append(samplex.SAMPLEX(d, value, ep, None, None, "SUBSET"))
            mechanism_instances[-1].name = mk
        elif mk == 'ADDITIVE':
            mechanism_instances.append(additive.ADDITIVE(d, value, ep))
            mechanism_instances[-1].name = mk
    return mechanism_instances


def Comb(n, k):
    b = D(1.0)
    for i in range(0, k):
        b = b*D((n-i)/(i+1))
    return b

__name__=["Comb"]
