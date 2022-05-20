__author__ = 'westbrick'
# class for binary randomized response

import numpy as np
import numpy.random as r
import time
import math
import utils

class SAMPLEX:
    name = 'SAMPLEX'
    ep = 0.0    # privacy budget epsilon

    d = 0       # number of candidates
    values = None   # value list in ranked voting profile

    # for encode
    base = None # base categorical randomizer
    weights = None # sampling weights for each ranked value
    bias = 0.0 # biased added to all values

    # records for consuming time
    clienttime = 0.0
    recordtime = 0.0
    servertime = 0.0

    def __init__(self, d, values, ep, bias=None, weights=None, base="BRR"):
        self.ep = ep
        self.d = d
        self.values = np.array(values)
        self.bias = bias
        self.weights = weights
        self.base = base
        self.k = 0

        # records for consuming time
        self.clienttime = 0.0
        self.recordtime = 0.0
        self.servertime = 0.0


        self.__setparams()

    def __setparams(self):
        if self.bias == None:
            self.bias = self.values[self.d-1]
            #self.bias = np.median(self.values)
        print("  samplex: bias", self.bias)
        if self.weights == None:
            self.weights = np.absolute(self.values-self.bias)
            self.weights = self.weights/np.sum(self.weights)


    def randomizer(self, permutation, profile=None, base=None):
        if base == None:
            base = self.base
        tstart = time.process_time()
        p = r.random()
        ri = 0
        #print(probs, p)
        while p - self.weights[ri] >= 0.0:
            p -= self.weights[ri]
            ri += 1
        
        # map ri in ranked list to an index in the profile
        si = 0
        while permutation[si] != ri:
            si += 1

        secrets = [0]*self.d
        secrets[si] = 1
        pub = np.array(secrets)

        if base == "BRR":
            trate = np.exp(self.ep/2)/(np.exp(self.ep/2)+1)
            frate = 1.0/(np.exp(self.ep/2)+1)
            ps = r.random(self.d)
            for i in range(0, self.d):
                if secrets[i] > 0:
                    if ps[i] < trate:
                        pub[i] = 1
                    else:
                        pub[i] = 0
                else:
                    if ps[i] < frate:
                        pub[i] = 1
                    else:
                        pub[i] = 0
            pub = (pub-np.array([frate]*self.d))/(trate-frate)
        elif base == "LAPLACE":
            pub = secrets + np.array(r.laplace(0.0, 2/self.ep, self.d))
        elif base == "PIECEWISE":
            # self.weights MUST BE [1/d]*d
            # Randomize self.values[ri] with piesewise mechanism
            v = self.values[ri]
            # normalize to [-1,1]
            nv = (v-self.values[-1])/(self.values[0]-self.values[-1])
            nv = 2*nv-1.0
            # randomize
            C = (np.exp(self.ep/2)+1)/(np.exp(self.ep/2)-1)
            left = (C+1)*nv/2-(C-1)/2
            right = left+C-1
            z = None
            if r.random() < np.exp(self.ep/2)/(np.exp(self.ep/2)+1):
                z = r.random()*(right-left)+left
            else:
                z = r.random()*(C-right+left+C)
                if z > left+C:
                    z = z+(right-left)-C
                else:
                    z = z-C
            # reverse normalize to [0, self.values[0]-self.values[-1]]
            z = (z+1.0)*(self.values[0]-self.values[-1])/2
            pub = np.zeros(self.d, dtype=float)+self.values[-1]
            # debias due to dimension sampling
            pub[si] += z*self.d
        else:
            # using k-subset mechanism
            #print("bug", self.ep, np.exp(self.ep), self.d)
            k = math.ceil(self.d/(np.exp(self.ep)+1))
            self.k = k
            trate = (k*np.exp(self.ep))/(k*np.exp(self.ep)+self.d-k)
            frate = (k*np.exp(self.ep)*(k-1)/(self.d-1)+(self.d-k)*k/(self.d-1))/(k*np.exp(self.ep)+self.d-k)
            pub = []
            p = r.random()
            l = list(range(0, self.d))
            #print(p, self.d, self.ep, k, trate, frate)
            del(l[si])
            if p <= trate:
                pub.append(si)
                pub.extend(utils.reservoirsample(l, k-1))
            else:
                pub.extend(utils.reservoirsample(l, k))
            pub = [int(i in pub) for i in range(0, self.d)]
            pub = (pub-np.array([frate]*self.d))/(trate-frate)
        self.clienttime += time.process_time()-tstart
        
        # now pub is an unbised estimator of secrets
        tstart = time.process_time()
        fs = pub*(self.values[ri]-self.bias)/self.weights[ri]+np.array([self.bias]*self.d)
        if base in ["PIECEWISE"]:
            fs = pub
        #print("runtime", self.weights[ri], self.values[ri], fs)
        self.servertime += time.process_time()-tstart
        return fs


    def decoder(self, sums, n):
        # debias hits but without projecting to simplex
        #print('rates', self.trate, self.frate)
        tstart = time.process_time()
        fs = sums
        self.servertime += time.process_time()-tstart
        return fs

    def disguiseViews(self, n, permutation, profile=None, base=None):
        # create n disguised private views follows permutation
        if base == None:
            base = self.base
        fsvalues = [[]]*self.d
        for ri in range(0, self.d):
            si = 0
            while permutation[si] != ri:
                si += 1
            secrets = [0] * self.d
            secrets[si] = 1
            pub = np.array(secrets)
            if base == "BRR":
                trate = np.exp(self.ep/2)/(np.exp(self.ep/2)+1)
                frate = 1.0/(np.exp(self.ep/2)+1)
                pub[permutation[0]] = 1
                pub[permutation[-1]] = 0
                pub = (pub - np.array([frate] * self.d)) / (trate - frate)
            elif base == "LAPLACE":
                alpha = 4.0
                pub[permutation[0]] = alpha / self.ep
                pub[permutation[-1]] = -alpha / self.ep
                # for i in range(0, seld.d):
                #    pub[permutation[i]] = (-1 + 2 * (self.d - 1 - i) / (self.d - 1)) * alpha / self.ep
            elif base == "PIECEWISE":
                C = (np.exp(self.ep/2)+1)/(np.exp(self.ep/2)-1)
                pub = np.zeros(self.d, dtype=float)+self.values[-1]
                pub[permutation[0]] += self.d*(C+1.0)*(self.values[0]-self.values[-1])/2
                pub[permutation[-1]] += self.d*(-C+1.0)*(self.values[0]-self.values[-1])/2
            else:
                # using k-subset mechanism
                #print("bug", self.ep, np.exp(self.ep), self.d)
                k = math.ceil(self.d/(np.exp(self.ep)+1))
                trate = (k*np.exp(self.ep))/(k*np.exp(self.ep)+self.d-k)
                frate = (k*np.exp(self.ep)*(k-1)/(self.d-1)+(self.d-k)*k/(self.d-1))/(k*np.exp(self.ep)+self.d-k)
                pub = [0.0]*self.d
                for i in range(0, self.k):
                    pub[permutation[i]] = 1.0
                #pub = [int(pub[i]) for i in range(0, self.d)]
                pub = (np.array(pub)-frate)/(trate-frate)
            fsvalues[ri] = pub * (self.values[ri] - self.bias) / (self.weights[ri]+0.0000000000000000000000001) + np.array([self.bias] * self.d)
            if base in ["PIECEWISE"]:
                fsvalues[ri] = pub

        maxri = -1
        maxdiff = 0.0
        for ri in range(0, self.d):
            if maxri < 0 or fsvalues[ri][permutation[0]]-fsvalues[ri][permutation[-1]] >= maxdiff:
                maxri = ri
                maxdiff = fsvalues[ri][permutation[0]]-fsvalues[ri][permutation[-1]]
        return [fsvalues[maxri]]*n


    def bound(self, n, tfs=None):
        # compute theoretical squared l2-norm error bound
        return 0.0
