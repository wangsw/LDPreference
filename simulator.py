__author__ = 'westbrick'
# simulator for local private discrete distribution estimation

import json
import numpy as np
import scipy as sp
import numpy.random as r

import utils
from distance import *
from datetime import datetime, date, time
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score

class Simulator:
    n = 0  # number of providers
    ds = None  # a list of domain size
    values = None  # a list of ranked profile values
    eps = None  # a list of privacy budget epsilons
    profiles = None  # voting profiles
    results = {}  # dict to record simulation settings and results
    mechanisms = None
    na = None # numbers of data amplification attack
    nd = None # numbers of view disguise attack
    repeat = 100  # repeat time for each simulation

    def init(self, n, ds, values, eps, repeat, mechanisms, permutations=None, na=None, nd=None):
        self.n = n
        self.ds = ds
        self.values = values
        self.eps = eps
        self.permutations = permutations
        self.profiles = None
        self.repeat = repeat
        self.mechanisms = mechanisms
        self.na = na
        self.nd = nd
        self.results['n'] = self.n
        self.results['ds'] = self.ds
        self.results['values'] = self.values
        self.results['eps'] = self.eps
        self.results['permutations'] = self.permutations
        self.results['profiles'] = self.profiles
        self.results['repeat'] = self.repeat
        self.results['na'] = self.na
        self.results['nd'] = self.nd
        self.results['mechanisms'] = self.mechanisms
        self.results['all'] = []
        #print("init", na, nd)
        for di in range(0, len(ds)):
            d = ds[di]
            self.results['di' + str(di)] = {}
            for ep in eps:
                self.results['di' + str(di)]['ep'+str(ep)] = {}
                self.results['di' + str(di)]['ep' + str(ep)]['histograms'] = [None] * self.repeat
                for cna in na:
                    for cnd in nd:
                        self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)] = {}
                        self.allresults = []
                        for mk in self.mechanisms:
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['estimators_'+mk] = [None]*self.repeat
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_l2_' + mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_l1_' + mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_linf_' + mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_acc_' + mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_exloss_' + mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_kt_' + mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_ndcg_' + mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_l2_'+mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_l1_'+mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_linf_'+mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_acc_'+mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_exloss_'+mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_kt_'+mk] = 0.0
                            self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_ndcg_'+mk] = 0.0


    def simulate(self):
        for di in range(0, len(self.ds)):
            d = self.ds[di]
            print("self.values", self.values)
            value = self.values[di]
            for ep in self.eps:
                # initialize mechanisms
                mechanism_instances = utils.initmechanisms(self.mechanisms, d, value, ep)
                # continue
                #print('d=', d, ', value=', value, ', epsilon=', ep, ', pa=', pa, ', pd=', pd, ', starts')
                print('d=', d, ', value=', value, ', epsilon=', ep, ', starts')
                for rt in range(0, self.repeat):
                    #for apa in pa:
                    #    for apd in pd:

                    permutations = None
                    if self.permutations == None:
                        honestpermutations = utils.randomPermutations(self.n+np.max(self.na), d, value)
                        # generate random votes as data amplification attack
                        aapermutations = utils.randomPermutations(np.max(self.na), d, value, [1.0]*d, repeat=np.max(self.na))
                        permutations = np.concatenate((honestpermutations, aapermutations), axis=0)
                    profiles = np.array([
                            [value[permutations[j][i]] for i in range(0, d)] 
                            for j in range(0, self.n+2*np.max(self.na))])
                    #h = profiles[0:self.n+np.max(self.na)].sum(axis=0)
                    # print("profiles & h", profiles, h)
                    #self.results['di' + str(di)]['ep'+str(ep)]['histograms'][rt] = h.tolist()
                    #nh = h/(self.n+np.max(self.na))
                    for mk in mechanism_instances:
                        # print('mechanism', mk.name)
                        # randomizer and decoder
                        pubs = utils.rawDistributor(self.n+2*np.max(self.na), permutations, mk)
                        for cna in self.na:
                            h = profiles[0:self.n+cna].sum(axis=0)
                            # print("profiles & h", profiles, h)
                            self.results['di'+str(di)]['ep' + str(ep)]['histograms'][rt] = h.tolist()
                            nh = h/(self.n+cna)

                            aeh = pubs[0:self.n].sum(axis=0)
                            aeh += pubs[self.n+np.max(self.na):self.n+np.max(self.na)+cna].sum(axis=0)

                            honestrank = np.array(rankdata(aeh, method='ordinal')-1).astype(int)
                            #print("honestrank:", aeh, honestrank)
                            manupulationrank = [0]*d
                            for i in range(0, d):
                                if honestrank[i] == 0:
                                    honestrank[i] = d-1
                                elif honestrank[i] == 1:
                                    honestrank[i] = 0
                                else:
                                    honestrank[i] -= 1
                                manupulationrank[honestrank[i]] = i
                            #print('manupulationrank', manupulationrank, honestrank)
                            disguisepubs = np.array(mk.disguiseViews(np.max(self.nd), manupulationrank))
                            for cnd in self.nd:
                                eh = aeh + disguisepubs[0:cnd].sum(axis=0)
                                neh = eh / (self.n + cna + cnd)

                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['estimators_'+mk.name][rt] = eh.tolist()

                                #print(nh, ws, nwh)
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_l2_'+mk.name] += l2norm(nh[0:d], neh[0:d])
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_l1_'+mk.name] += l1norm(nh[0:d], neh[0:d])
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_linf_'+mk.name] += infnorm(nh[0:d], neh[0:d])
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_acc_'+mk.name] += accuracyargmax(nh[0:d], neh[0:d])
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_exloss_'+mk.name] += excessloss(nh[0:d], neh[0:d])
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_kt_'+mk.name] += sp.stats.kendalltau(nh[0:d], neh[0:d])[0]
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_ndcg_'+mk.name] += ndcg_score([nh[0:d]], [neh[0:d]])

                                npeh = utils.projector(neh/np.sum(value))*np.sum(value)
                                #print(nh, npeh, l2norm(nh, npeh))

                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_l2_'+mk.name] += l2norm(nh[0:d], npeh[0:d])
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_l1_'+mk.name] += l1norm(nh[0:d], npeh[0:d])
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_linf_'+mk.name] += infnorm(nh[0:d], npeh[0:d])
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_acc_'+mk.name] += accuracyargmax(nh[0:d], npeh[0:d])
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_exloss_'+mk.name] += excessloss(nh[0:d], npeh[0:d])
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_kt_'+mk.name] += sp.stats.kendalltau(nh[0:d], npeh[0:d])[0]
                                self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_ndcg_'+mk.name] += ndcg_score([nh[0:d]], [npeh[0:d]])

                        if self.repeat < 10:
                            print("time", mk.name, mk.clienttime, mk.recordtime, mk.servertime, self.n, d, ep)
                        #print("h eh", h.tolist(), eh.tolist(), mk.name)


                        """
                        print('  ' + str(rt) + 'core_raw_' + mk.name,
                              self.results['di' + str(di)]['ep' + str(ep)]['core_raw_mean_l2_' + mk.name] / (rt + 1),
                              self.results['di' + str(di)]['ep' + str(ep)]['core_raw_mean_l1_' + mk.name] / (rt + 1),
                              self.results['di' + str(di)]['ep' + str(ep)]['core_raw_mean_linf_' + mk.name] / (rt + 1))
                        print('  ' + str(rt) + 'core_prj_' + mk.name,
                              self.results['di' + str(di)]['ep' + str(ep)]['core_mean_l2_' + mk.name] / (rt + 1),
                              self.results['di' + str(di)]['ep' + str(ep)]['core_mean_l1_' + mk.name] / (rt + 1),
                              self.results['di' + str(di)]['ep' + str(ep)]['core_mean_linf_' + mk.name] / (rt + 1))
                        """
                    # print('iteration=', rt, ' ends')


                #print(self.na, self.nd)
                for cna in self.na:
                    for cnd in self.nd:
                        print("\n  Data amplification attacks:", cna, ", View disguise attacks:", cnd)
                        for mk in mechanism_instances:
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_l2_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_l1_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_linf_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_acc_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_exloss_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_kt_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_ndcg_'+mk.name] /= self.repeat

                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_l2_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_l1_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_linf_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_acc_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_exloss_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_kt_'+mk.name] /= self.repeat
                            self.results['di' + str(di)]['ep'+str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_ndcg_'+mk.name] /= self.repeat

                            self.results['all'].append(
                                [
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_l2_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_l1_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_linf_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_acc_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_exloss_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_kt_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_raw_mean_ndcg_' + mk.name]
                                ]
                            )
                            self.results['all'].append(
                                [
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_l2_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_l1_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_linf_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_acc_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_exloss_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_kt_' + mk.name],
                                    self.results['di' + str(di)]['ep' + str(ep)]['aa'+str(cna)+'da'+str(cnd)]['core_mean_ndcg_' + mk.name],
                                ]
                            )
                            print('core_raw_' + mk.name, self.results['all'][-2])

                            print('core_prj_' + mk.name, self.results['all'][-1])

                    # print
                # print('d=', d, ', epsilon=', ep, ', ends')


    def write(self, filename):
        with open(datetime.now().isoformat().replace(':', '_')+'-'+filename, 'w') as outfile:
            json.dump(self.results, outfile)


    def read(self, filename):
        with open(filename, 'r') as data_file:
            self.results = json.load(data_file)
        self.n = self.results['n']
        self.ds = self.results['ds']
        self.values = np.array(self.results['values'])
        self.eps = self.results['eps']
        self.repeat = self.results['repeat']













