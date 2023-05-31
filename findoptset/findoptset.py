import sys
sys.path.append('./utils/')
from steq import StEq
from jsonio import load,save
from graph import sort_topology
from base import BaseSet
from tensor import dfTotensor,concat
from neuralnet import regularizer,EarlyStopping,NonLinearF

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from copy import deepcopy
from functools import partial

class OptSet(BaseSet):
    """
    estimate fS'
        1. shffle @X_do* and regenerate its descentants
        2. train Y = f_S'(X_S',do(X_M)) in the regenereted samples
    estimate h*(S')
        1. replace M1,M2,... by J1,J2,...
           regenerate their descentants (topolygically)
        2. calculate Y_hat = f_S'(X_S',do(X_M)), where X_S',X_M~P(J)
        3. compute neg mse loss and optimize over Js
    """
    def __init__(self,trainsplit,path,parser,need_norm):
        super().__init__(trainsplit,path,parser,need_norm)
        self.sort = partial(sort_topology,order=self.parser['topology'])
        # if debugging mode
        self.debug = '/data/simulation/' in self.path
        self._structual_eqs()
    
    def _structual_eqs(self,):
        '''
        estimate st-eqs for data regeneration (De(doStar), De(X_M))
        '''
        verbose = False
        keys = [('De_dostar','PA_De_dostar'),('De_M_intervened','PA_De_M_intervened')]
        self.steq = dict()
        
        for key in keys:
            for node in self.parser[key[0]]:
                if node in self.steq.keys():
                    continue
                pa = self.parser[key[1]][node]
                net = NonLinearF(in_dim=len(pa),baseinit=True,hidden=max(1,len(pa)//2),verbose=verbose)
                net.fit(concat(self.trainData,pa),self.trainData[node],
                       opt='Adam',num_iters=5000,lr=0.01,earlystop=True,log=False)
                self.steq[node] = net
        
        # example: you can use the gt-steq for debugging
        if self.debug:
            self.steq['X1'] = StEq(nonlin=torch.sigmoid, lin=[1.13, 1.51, -0.58, 0.63])
            self.steq['X2'] = StEq(nonlin=torch.tanh, lin=[1.47, -0.64])
            self.steq['X3'] = StEq(nonlin=torch.tanh, lin=[-0.95])
            self.steq['X4'] = StEq(nonlin=torch.sigmoid, lin=[1.42, 0.99, 0.72])

    
    def _f_S_prime(self,):
        '''
        estimate the fSprime predictor E[Y|Sprime,do(X*)]
        '''
        verbose = False
        self.f_S_prime = NonLinearF(in_dim=len(self.S_prime)+len(self.M), 
                        baseinit=True, hidden=12, verbose=verbose)

        # shuffle doStar
        shufDF = deepcopy(self.trainDF)
        for node in self.doStar:
            shufDF.loc[:, node] = shufDF.loc[:, node].sample(frac=1, random_state=123).values

        # regenerate De(doStar)
        for node in self.sort(self.parser['De_dostar']):
            pa = self.parser['PA_De_dostar'][node]
            shufDF.loc[:,node] = self.steq[node].predict(shufDF[pa])

        # fit in the interventional distribution
        self.f_S_prime.fit(shufDF[self.S_prime + self.M], shufDF[['Y']],
            opt='Adam', num_iters=10000, lr=0.02, earlystop=(not self.debug), log=False)

    def _hstar(self,):
        '''
        estimate worst-case risk of the subset
        '''
        verbose = False
        interval = 10 # save h* every interval (iters)
        num_iters = 10000
        lr = 0.02
        
        # J: PA_M -> X_M
        fJ = dict()
        fJparams = list()

        for node in self.parser['X_M']:
            pa = self.parser['PA_M'][node]
            # patch
            if self.debug:
                hnum = 2 if node=='X9' else 3
            else:
                hnum = max(2,len(pa)//2)
            net = NonLinearF(in_dim=len(pa), baseinit=True, hidden=hnum, out_dim=1)
            fJ[node] = net
            fJparams += list(net.parameters())
            
        optimizer = torch.optim.Adam(fJparams, lr=lr)
        earlystoper = EarlyStopping(patience=100, min_delta=0.01, verbose=False)
        
        loss_func = nn.MSELoss()
        log = list()
        doData = deepcopy(self.trainData)
        
        for itera in range(num_iters + 1):
            # do(M=J(PA_M)) and regenerate De(M)
            for node in self.sort(self.parser['X_M'] + self.parser['De_M_intervened']):
                if node in self.parser['X_M']:
                    pa = self.parser['PA_M'][node]
                    #print('do({}=J{})'.format(node,pa))
                    doData[node] = fJ[node](concat(doData,pa))
                else:
                    pa = self.parser['PA_De_M_intervened'][node]
                    #print('pred {} <- {}'.format(node,pa))
                    doData[node] = self.steq[node](concat(doData,pa))

            # neg MSE loss
            predY = self.f_S_prime(concat(doData, self.S_prime + self.M))
            mse = loss_func(doData['Y'], predY)
            wnorm = sum([regularizer(fJ[key]) for key in fJ.keys()])
            loss = - mse + wnorm

            if itera % (num_iters // 5) == 0:
                print('iteration: {:d}, mse: {:.3f}, wnorm: {:.3f}, lr:{:.4f}'.format(int(itera), float(mse),
                    float(wnorm),optimizer.state_dict()['param_groups'][0]['lr'])) if verbose else None

            if itera%interval == 0:
                log.append(float(mse.detach()))
                earlystoper(log[-1])
                if earlystoper.early_stop:
                    break

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        return log + [log[-1]] * (num_iters // interval + 1 - len(log)) # make sure all log has the same length
    
    def _test(self, ):
        log = list()
        for Data in self.testDatas:
            Y = Data['Y']
            Y_pred = self.f_S_prime(concat(Data, self.S_prime + self.M))

            mse = torch.mean((Y - Y_pred) ** 2).item()
            log.append(mse)

        return log
    
    def estimate(self,S_prime):
        '''
        entry function
        S_prime is a list of string, e.g., ['X1','X3'], presenting the subset you want to estimate
        '''
        for node in self.M:
            assert node not in S_prime, 'S_prime contain a mutable variable {}.'.format(node)
        self.S_prime = S_prime
        print('Train_split: {}, S_prime: {}'.format(self.trainsplit, ','.join(self.S_prime)))
        
        self._f_S_prime()
        hlog = self._hstar()
        testlog = self._test()
        
        return hlog,testlog
    
def train(split,OptSet,S_prime_all,path,parser,need_norm):
    record = dict()
    for S_prime in S_prime_all:
        save_name = ','.join(S_prime) if len(S_prime) > 0 else 'empty'
        record[save_name] = dict()
        record[save_name]['hstar'] = list()
        record[save_name]['mse'] = list()
        
    # TODO: allow hyper-params input
    opt = OptSet(split,path,parser,need_norm)
    for S_prime in S_prime_all:
        save_name = ','.join(S_prime) if len(S_prime) > 0 else 'empty'
        hlog, testlog = opt.estimate(S_prime)

        record[save_name]['hstar'].append(hlog)
        record[save_name]['mse'].append(testlog)

    return record