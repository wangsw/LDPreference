__author__ = 'westbrick'
# class for additive mechanism

import numpy as np
import numpy.random as r
import time
import math
import utils

class ADDITIVE:
    name = 'ADDITIVE'
    ep = 0.0    # privacy budget epsilon

    d = 0       # number of candidates
    values = None   # value list in ranked voting profile

    # for encode
    weights = None # sampling weights for each ranked value
    presents = None # present probability in the output of each ranked value
    mins = 0.0 # min total values for a output subset 
    maxs = 0.0 # max total values for a output subset

    errorbounds = 0.0 # mean squared error


    # records for consuming time
    clienttime = 0.0
    recordtime = 0.0
    servertime = 0.0

    def __init__(self, d, values, ep, bias=None, k=None):
        self.ep = ep
        self.d = d
        self.values = np.array(values)
        self.k = k

        # records for consuming time
        self.clienttime = 0.0
        self.recordtime = 0.0
        self.servertime = 0.0


        self.__setparams()

    def __setparams(self):
        # choice of k
        self.k, self.mins, self.maxs, self.weights, self.presents, self.errorbounds = self.argk(self.d, self.values, self.ep, self.k)
        print("  additive: optimal k", self.k)


    @staticmethod
    def argk(d, values, ep, k=None):
        # select optimal choice of k according to domain size, ranked values and privacy budget
        ms = {}
        for i in range(1, d):
            ms[i] = {}
            ms[i]['mins'] = np.sum(values[-i:])
            ms[i]['maxs'] = np.sum(values[0:i])
            ms[i]['weights'] = (values-ms[i]['mins']/i)*(np.exp(ep)-1)/(ms[i]['maxs']-ms[i]['mins'])+1/i
            totalweights = np.sum(ms[i]['weights'])
            ms[i]['presents'] = ms[i]['weights']*(d-i)/((d-1)*totalweights) + (i-1)/(d-1)
            ms[i]['errorbounds'] = np.sum(ms[i]['presents']*(1-ms[i]['presents']))*np.square(totalweights*(ms[i]['maxs']-ms[i]['mins'])*(d-1))/np.square((d-i)*(np.exp(ep)-1))
            
            #ak = (np.sum(values)*(np.exp(ep)-1)-np.exp(ep)*ms[i]['mins']*d/i+ms[i]['maxs']*d/i)*(d-1)/((d-i)*(np.exp(ep)-1))
            #bk = (np.sum(values)*(np.exp(ep)-1)*(i-1)/(d-1)-np.exp(ep)*ms[i]['mins']+ms[i]['maxs'])*(d-1)/((d-i)*(np.exp(ep)-1))
            #bounds = ak*ak*np.sum((values+bk)/ak*(1-(values+bk)/ak))
            #print("Both right bounds:", ms[i]['errorbounds'], bounds)

        ms[d-1]['errorbounds'] *= 1.02
        print("  errorbounds(k): ", [ms[i]['errorbounds'] for i in range(1, d)])
        if k == None:
            # find the k with minimum error bounds
            minbound = None
            for i in range(1, d):
                if k == None or ms[i]['errorbounds'] < minbound:
                    k = i
                    minbound = ms[k]['errorbounds']
        return k, ms[k]['mins'], ms[k]['maxs'], ms[k]['weights'], ms[k]['presents'], ms[k]['errorbounds']



    def randomizer(self, permutation, profile=None):
        tstart = time.process_time()
        # recursively choose a maximum element
        ranki = [0]*self.d
        for j in range(0, self.d):
            ranki[permutation[j]] = j
        pub = [0]*self.d

        ci = 0 # current starting index under consideration 
        csum = 0.0 # current sum of weights of element considered
        for i in range(0, self.k):
            # select the top rank element according to probability design
            k = self.k - i
            weights = self.weights + csum/k
            totalweights = np.sum(weights[ci:])
            #print(weights, totalweights, ci, csum, k, float(utils.Comb(self.d-ci-0, k-1)), self.d-ci-0-1)
            #maxpresents = [weights[ci+j]*(self.d-ci-self.k+i)/((self.d-ci-1)*np.sum(np.array(weights[ci+j:]))) for j in range(0, self.d-ci)]
            maxpresents = np.array([float(utils.Comb(self.d-ci-j-1, k-1))*(weights[ci+j]+(np.sum(weights[ci+j:])-weights[ci+j])*(k-1)/(self.d-ci-j-1+0.0000000000000000000000000001)) for j in range(0, self.d-ci-k+1)])
            maxpresents = maxpresents/np.sum(maxpresents)
            p = r.random()
            #print("maxpresents", maxpresents, p)
            ri = 0
            while p - maxpresents[ri] >= 0.0:
                p -= maxpresents[ri]
                ri += 1
            si = ci+ri
            pub[ranki[si]] = 1

            ci = si+1
            csum += self.weights[si]
        self.clienttime += time.process_time()-tstart

        # from pub:subset to an estimator
        tstart = time.process_time()
        fs = (np.array(pub)-(self.k-1)/(self.d-1))*np.sum(self.weights)*(self.d-1)/(self.d-self.k)
        fs -= (1-self.mins*(np.exp(self.ep)-1)/(self.maxs-self.mins))/self.k
        fs *= (self.maxs-self.mins)/(np.exp(self.ep)-1)
        ak = (np.sum(self.values)*(np.exp(self.ep)-1)-np.exp(self.ep)*self.mins*self.d/self.k+self.maxs*self.d/self.k)*(self.d-1)/((self.d-self.k)*(np.exp(self.ep)-1))
        bk = (np.sum(self.values)*(np.exp(self.ep)-1)*(self.k-1)/(self.d-1)-np.exp(self.ep)*self.mins+self.maxs)*(self.d-1)/((self.d-self.k)*(np.exp(self.ep)-1))
        #print("Both Right:", fs, np.array(pub)*ak-bk)
        self.servertime += time.process_time()-tstart
        

        return fs


    def decoder(self, sums, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        tstart = time.process_time()
        fs = sums
        self.servertime += time.process_time()-tstart
        return fs

    def disguiseViews(self, n, permutation, profile=None):
        # create n disguised private views follows permutation
        pub = [0.0]*self.d
        for i in range(0, self.k):
            pub[permutation[i]] = 1.0
        # subset to estimator
        fs = (np.array(pub) - (self.k - 1) / (self.d - 1)) * np.sum(self.weights) * (self.d - 1) / (self.d - self.k)
        fs -= (1 - self.mins * (np.exp(self.ep) - 1) / (self.maxs - self.mins)) / self.k
        fs *= (self.maxs - self.mins) / (np.exp(self.ep) - 1)
        return [fs]*n


    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        return self.errorbounds/n
