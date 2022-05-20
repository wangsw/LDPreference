__author__ = 'westbrick'
# class for binary randomized response

import numpy as np
import numpy.random as r
import time

class LAPLACE:
    name = 'LAPLACE'
    ep = 0.0    # privacy budget epsilon

    d = 0       # number of candidates
    values = None   # value list in ranked voting profile

    # for encode
    #weights = None # sampling weights for each ranked value
    bias = 0.0 # biased added to all values
    delta = None # sensitivity


    # records for consuming time
    clienttime = 0.0
    recordtime = 0.0
    servertime = 0.0

    def __init__(self, d, values, ep, bias=None):
        self.ep = ep
        self.d = d
        self.values = np.array(values)
        self.bias = bias
        #self.weights = np.array(weights)

        # records for consuming time
        self.clienttime = 0.0
        self.recordtime = 0.0
        self.servertime = 0.0


        self.__setparams()

    def __setparams(self):
        if self.bias == None:
            self.bias = 0.0
        #if self.weights == None:
        #    self.weights = [1.0/self.d for i in range(0, self.d)]
        # To Fix
        rawdelta = 2.0*np.sum(np.absolute(self.values-np.array([self.bias]*self.d)))
        #print("values", self.values)
        self.delta = np.sum(np.absolute(self.values-np.flip(self.values, axis=0)))
        print("  delta", rawdelta, self.delta)

    def randomizer(self, permutation, profile=None):
        if profile == None:
            profile = [self.values[permutation[i]] for i in range(0, self.d)]
        tstart = time.process_time()
        pub = np.array(profile)-np.array([self.bias]*self.d)
        pub += np.array(r.laplace(0.0, self.delta/self.ep, size=self.d))
        self.clienttime += time.process_time()-tstart
        return pub


    def decoder(self, sums, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        tstart = time.process_time()
        fs = sums
        self.servertime += time.process_time()-tstart
        return fs

    def disguiseViews(self, n, permutation, profile=None):
        # create n disguised private views follows permutation
        #if profile == None:
        #    profile = [self.values[permutation[i]] for i in range(0, self.d)]
        pub = np.array([0.0]*self.d)
        alpha = np.log(1/0.05)*self.delta
        pub[permutation[0]] = alpha/self.ep+self.values[0]
        pub[permutation[-1]] = -alpha/self.ep+self.values[-1]
        #for i in range(0, seld.d):
        #    pub[permutation[i]] = (-1 + 2 * (self.d - 1 - i) / (self.d - 1)) * alpha / self.ep
        return [pub]*n


    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        return 2*np.square(self.delta/self.ep)*self.d/n
