import sys
sys.path.append('../utils/')
from tools import dfTotensor,_print,concat,NonLinearF,EarlyStopping,train
from base import BaseSet
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from copy import deepcopy
from itertools import combinations
import random
import multiprocess as mp
from functools import partial
from utils import stack,save,spearman,load,spearman
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
import shutil


class OptSet(BaseSet):
    r""" ADNI25Nodes, BK3.1, SP-1
    1. _structual_eqs
      - X5 = f5(X6,X11,X19,Y)
      - X4 = f4(X3,X5,X6,X12,X22)
      - X9 = f9(X5,X8,X23)
      - X14 = f14(X9,X15,X23)
      - X13 = f13(X2,X14,X16)
      - X17 = f17(X19)
    2. _f_S_prime
      - shffle X_do*={X4,X9,X13,X14,X17,X19} and regenerate X5 by f5
      - train Y = f_S'(X_S',do(X_M)) in the regenereted samples, where X_M={X18,X19,X20}
    3. _hstar
      - replace X18 by fJ18(X21,X22)
        replace X19 by fJ19(X25,Y)
        replace X20 by fJ20(X21)
      - regenerate X_i by f(PA_i) for X_i is a descent of X18,X19,X20 in the induced graph
          - X5 = f5(X6,X11,X19,Y)
          - X4 = f4(X3,X5,X6,X12,X22)
          - X9 = f9(X5,X8,X23)
          - X14 = f14(X9,X15,X23)
          - X13 = f13(X2,X14,X16)
          - X17 = f17(X19)
      - calculate Y_hat = f_S'(X_S',do(X_M)), where X_S',X_M~P(J_theta)
        compute negMSELoss -||Y-Y_hat|| and optimize over \theta
    """

    def __init__(self, trainsplit, path, params, need_norm):
        super().__init__(trainsplit, path, need_norm)
        self.S = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12',
                  'X13', 'X14', 'X15', 'X16', 'X17', 'X21', 'X22', 'X23', 'X24', 'X25']
        self.M = ['X18', 'X19', 'X20']
        self.doStar = ['X4', 'X9', 'X13', 'X14', 'X17', 'X19']
        self.params = params
        self._structual_eqs()

    def _structual_eqs(self, ):
        verbose = False
        earlystop = False

        self.f5 = NonLinearF(in_dim=4, baseinit=True, hidden=1, verbose=verbose)
        self.f5.fit(concat(self.trainData, ['X6', 'X11', 'X19', 'Y']), self.trainData['X5'],
                    opt='Adam', num_iters=5000, lr=0.01, earlystop=earlystop, log=False)

        self.f4 = NonLinearF(in_dim=5, baseinit=True, hidden=1, verbose=verbose)
        self.f4.fit(concat(self.trainData, ['X3', 'X5', 'X6', 'X12', 'X22']), self.trainData['X4'],
                    opt='Adam', num_iters=5000, lr=0.01, earlystop=earlystop, log=False)

        self.f9 = NonLinearF(in_dim=3, baseinit=True, hidden=1, verbose=verbose)
        self.f9.fit(concat(self.trainData, ['X5', 'X8', 'X23']), self.trainData['X9'],
                    opt='Adam', num_iters=5000, lr=0.01, earlystop=earlystop, log=False)

        self.f14 = NonLinearF(in_dim=3, baseinit=True, hidden=1, verbose=verbose)
        self.f14.fit(concat(self.trainData, ['X9', 'X15', 'X23']), self.trainData['X14'],
                     opt='Adam', num_iters=5000, lr=0.01, earlystop=earlystop, log=False)

        self.f13 = NonLinearF(in_dim=3, baseinit=True, hidden=1, verbose=verbose)
        self.f13.fit(concat(self.trainData, ['X2', 'X14', 'X16']), self.trainData['X13'],
                     opt='Adam', num_iters=5000, lr=0.01, earlystop=earlystop, log=False)

        self.f17 = NonLinearF(in_dim=1, baseinit=True, hidden=1, verbose=verbose)
        self.f17.fit(self.trainData['X19'], self.trainData['X17'],
                     opt='Adam', num_iters=5000, lr=0.01, earlystop=earlystop, log=False)

    def _f_S_prime(self, ):
        r"""
        Functs: - shuffle X_do*, regenerate X5, train Y=f_S_prime(X_S_prime,do X_M) in p*
        """
        verbose = self.params['verbose']
        earlystop = self.params['fSearly']
        weightcut = self.params['fScut']
        threshold = self.params['fScutthreshold']
        num_iters = self.params['fSIter']
        step = self.params['fSstep']
        lr = self.params['fSLr']
        gamma = self.params['fSgamma']
        opt = self.params['fSOpt']

        self.f_S_prime = NonLinearF(in_dim=len(self.S_prime) + len(self.M), baseinit=True, hidden=5, verbose=verbose)

        shufDF = deepcopy(self.trainDF)
        for node in self.doStar:
            shufDF.loc[:, node] = shufDF.loc[:, node].sample(frac=1, random_state=1234).values

        shufDF.loc[:, 'X5'] = self.f5.predict(shufDF[['X6', 'X11', 'X19', 'Y']])
        self.f_S_prime.fit(shufDF[self.S_prime + self.M], shufDF[['Y']],
                           opt=opt, num_iters=num_iters, lr=lr, step=step, gamma=gamma,
                           earlystop=earlystop, weightcut=weightcut, threshold=threshold, log=False)

    def _hstar(self, ):
        r"""
        Functs: - replace X18 by fJ8(X21,X22)
                          X19 by fJ19(X25,Y)
                          X20 by fJ20(X21)
                - regenerate
                         X5 = f5(X6,X11,X19,Y)
                         X4 = f4(X3,X5,X6,X12,X22)
                         X9 = f9(X5,X8,X23)
                         X14 = f14(X9,X15,X23)
                         X13 = f13(X2,X14,X16)
                         X17 = f17(X19)
                - neg mse loss
        """
        verbose = self.params['verbose']
        earlystop = self.params['fJearly']
        num_iters = self.params['fJIter']
        lr = self.params['fJLr']
        opt = self.params['fJopt']
        interval = 10

        if self.params['fJsplit']:
            fJ18 = NonLinearF(in_dim=2, baseinit=True, hidden=1, out_dim=1)
            fJ19 = NonLinearF(in_dim=2, baseinit=True, hidden=1, out_dim=1)
            fJ20 = NonLinearF(in_dim=1, baseinit=True, hidden=1, out_dim=1)
            thetaparams = list(fJ18.parameters()) + list(fJ19.parameters()) + list(fJ20.parameters())
        else:
            fJ = NonLinearF(in_dim=4, baseinit=True, hidden=len(self.M), out_dim=len(self.M))
            thetaparams = fJ.parameters()

        optimizer = torch.optim.SGD(thetaparams, lr=lr) if opt == 'SGD' else torch.optim.Adam(thetaparams, lr=lr)
        earlystoper = EarlyStopping(patience=100, min_delta=0.001, verbose=False)

        loss_func = nn.MSELoss()
        log = list()
        doData = deepcopy(self.trainData)

        for itera in range(num_iters + 1):
            # do(X18,X19,X20)
            if self.params['fJsplit']:
                doData['X18'] = fJ18(concat(doData, ['X21', 'X22']))
                doData['X19'] = fJ19(concat(doData, ['X25', 'Y']))
                doData['X20'] = fJ20(doData['X21'])
            else:
                predXM = fJ(concat(doData, ['X21', 'X22', 'X25', 'Y']))
                doData['X18'] = predXM[:, 0].unsqueeze(1)
                doData['X19'] = predXM[:, 1].unsqueeze(1)
                doData['X20'] = predXM[:, 2].unsqueeze(1)

            # regenerate descentants
            doData['X5'] = self.f5(concat(doData, ['X6', 'X11', 'X19', 'Y']))
            doData['X4'] = self.f4(concat(doData, ['X3', 'X5', 'X6', 'X12', 'X22']))
            doData['X9'] = self.f9(concat(doData, ['X5', 'X8', 'X23']))
            doData['X14'] = self.f14(concat(doData, ['X9', 'X15', 'X23']))
            doData['X13'] = self.f13(concat(doData, ['X2', 'X14', 'X16']))
            doData['X17'] = self.f17(concat(doData, ['X19']))
            # negative MSE loss
            predY = self.f_S_prime(concat(doData, self.S_prime + self.M))
            mse = loss_func(doData['Y'], predY)
            loss = - mse

            if itera % (num_iters // 5) == 0:
                print('iteration: {:d}, mse: {:.3f}, lr:{:.4f}'.format(int(itera), float(mse),
                                                                       optimizer.state_dict()['param_groups'][0][
                                                                           'lr'])) if verbose else None

            if itera % interval == 0:
                log.append(float(mse.detach()))
                if earlystop:
                    earlystoper(log[-1])
                if earlystoper.early_stop and earlystop:
                    break

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if self.params['fJsplit']:
            self.fJ18 = fJ18
            self.fJ19 = fJ19
            self.fJ20 = fJ20
        else:
            self.fJ = fJ
        return log + [log[-1]] * (num_iters // interval + 1 - len(log))

    def _test(self, ):
        log = list()
        for Data in self.testDatas:
            Y = Data['Y']
            Y_pred = self.f_S_prime(concat(Data, self.S_prime + self.M))

            mse = torch.mean((Y - Y_pred) ** 2).item()
            log.append(mse)

        return log

    def estimate(self, S_prime):
        S_prime = [] if len(S_prime) == 0 else S_prime.split(',')
        for node in self.M:
            assert node not in S_prime, 'Mistake: S_prime must not contain any X_M'

        self.S_prime = S_prime
        self._f_S_prime()
        hlog = self._hstar()

        testlog = self._test()

        return hlog, testlog

if __name__ == '__main__':
    for ind in [4,5]:
        S_prime_all = load('../25307_rep_subsets/formal{}.json'.format(ind))

        path = '/data/liumingzhou/CausallyInvariant_output/ICML23_ADNI25/FindOptSets/'
        splits = [s for s in os.listdir(path) if not s.endswith('.csv')]
        params = {'fSOpt': 'SGD', 'fSLr': 0.25, 'fSIter': 5000, 'fSstep': 4000, 'fSgamma': 0.4, 'fSearly': False,
                  'fScut': False, 'fScutthreshold': None,
                  'fJopt': 'SGD', 'fJLr': 0.4, 'fJIter': 12000, 'fJearly': True, 'fJsplit': True,
                  'verbose': False}

        # for code testing
        date_time = 'formal_{}'.format(ind)

        print('SubsetID: {}'.format(ind))
        train_par = partial(train, OptSet=OptSet, S_prime_all=S_prime_all, path=path, params=params)
        ctx = mp.get_context('spawn')
        start = time()
        with ctx.Pool(3) as pool:
            records = pool.map(train_par, splits)
            pool.close()
            pool.join()
        end = time()
        print('Time cost: {:.3f} h'.format((end - start) / 3600))

        record = stack(records)
        savepath = '/data/liumingzhou/CausallyInvariant_output/ICML23_ADNI25/Output'
        os.makedirs(os.path.join(savepath, date_time), exist_ok=True)
        save(record, os.path.join(savepath, date_time, 'record.json'))
        save(params, os.path.join(savepath, date_time, 'params.json'))
        spearman(record, filename=os.path.join(savepath, date_time, 'spearman.pdf'))
        shutil.copyfile('./opt_subset_selection.py', os.path.join(savepath, date_time, 'code.py'))
        print('save path: {}'.format(os.path.join(savepath, date_time)))