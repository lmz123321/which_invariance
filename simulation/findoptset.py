import sys
sys.path.append('./utils/')
from steq import StEq
from tensor import dfTotensor,concat
from neuralnet import regularizer,EarlyStopping,NonLinearF

from jsonio import save

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from copy import deepcopy
import random

# note that the node naming rule in the code is a bit different from Fig.4a in the paper
# we use the following code to transform
snodes = ['X2','X3','X4','X5','X6','X7','X10','X11']
snames = ['X{}'.format(ind+1) for ind in range(8)]
mnodes = ['M1','M2']
mnames = ['X9','X10']
renamer = dict(zip(snames+mnames+['Y','E'],snodes+mnodes+['Y','E'],))

class BaseSet():
    def __init__(self, trainsplit, path, need_norm):
        self.trainsplit = trainsplit

        # read, normalize, and convert to torch.tensor
        trainfilename = os.path.join(path, '{}.csv'.format(trainsplit))
        self.trainDF = pd.read_csv(trainfilename)
        self.trainDF.rename(columns=renamer,inplace=True)
        
        self.need_norm = need_norm
        if self.need_norm:
            for var in self.trainDF.columns.tolist():
                mean = self.trainDF[[var]].mean().values[0]
                std = self.trainDF[[var]].std().values[0]
                self.trainDF[[var]] = (self.trainDF[[var]] - mean) / std

        self.trainData = dict()
        for node in self.trainDF.columns.tolist():
            if node != 'E':
                self.trainData[node] = dfTotensor(self.trainDF[[node]])

        # test
        self.testDatas = list()
        testfolder = os.path.join(path, '{}'.format(trainsplit))
        for filename in os.listdir(testfolder):
            testDF = pd.read_csv(os.path.join(testfolder, filename))
            testDF.rename(columns=renamer,inplace=True)
            if self.need_norm:
                for var in self.trainDF.columns.tolist():
                    mean = testDF[[var]].mean().values[0]
                    std = testDF[[var]].std().values[0]
                    testDF[[var]] = (testDF[[var]] - mean) / std
            testData = dict()
            for node in testDF.columns.tolist():
                if node != 'E':
                    testData[node] = dfTotensor(testDF[[node]])
            self.testDatas.append(testData)
            
class OptSet(BaseSet):
    """
    estimate fS'
        1. shffle @X_do*={X3,X4,X5,M1,M2} and regenerate X2 by f2
        2. train Y = f_S'(X_S',do(X_M)) in the regenereted samples
    estimate h*(S')
        1. replace M1 by J1(Y)
           regenerate:
               X2 = f2(X6,X7,M1,Y), X4 = f4(X2), X3 = f3(X4,X11),X5 = f5(X3,X10,X11)
           replace M2 by J2(X4,X6,X11)
        2. calculate Y_hat = f_S'(X_S',do(X_M)), where X_S',X_M~P(J_theta)
        3. compute neg mse loss and optimize over J1,J2
    """

    def __init__(self, trainsplit, path, need_norm):
        super().__init__(trainsplit, path, need_norm)
        self.S = ['X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X10', 'X11']
        self.M = ['M1', 'M2']
        self.doStar = ['X3', 'X4', 'X5', 'M1', 'M2']
        self._structual_eqs()

    def _structual_eqs(self, ):
        self.f2 = StEq(nonlin=torch.sigmoid, lin=[1.13, 1.51, -0.58, 0.63])
        self.f3 = StEq(nonlin=torch.tanh, lin=[1.47, -0.64])
        self.f4 = StEq(nonlin=torch.tanh, lin=[-0.95])
        self.f5 = StEq(nonlin=torch.sigmoid, lin=[1.42, 0.99, 0.72])

    def _f_S_prime(self, ):
        r"""
        Functs: - sample from p* by shuffle X_do*, regenerate X2 by f2(X6,X7,M1,Y),
                - train Y=f_S_prime(X_S_prime,do X_M) in p*
        """
        verbose = False
        self.f_S_prime = NonLinearF(in_dim=len(self.S_prime) + len(self.M), baseinit=True, hidden=12, verbose=verbose)

        shufDF = deepcopy(self.trainDF)
        for node in self.doStar:
            shufDF.loc[:, node] = shufDF.loc[:, node].sample(frac=1, random_state=123).values
        shufDF.loc[:, 'X2'] = self.f2.predict(shufDF[['X6', 'X7', 'M1', 'Y']])

        self.f_S_prime.fit(shufDF[self.S_prime + self.M], shufDF[['Y']],
                           opt='Adam', num_iters=10000, lr=0.02, earlystop=False, log=False)
        return None

    def _hstar(self, ):
        r"""
        Functs: - M1 = J1(Y), then regenerate X2,X4,X3,X5, M2=J2(X4,X6,X11)
        """
        verbose = False
        interval = 10 # save h* every interval
        num_iters = 10000
        lr = 0.02
        fJ1 = NonLinearF(in_dim=1, baseinit=True, hidden=2, out_dim=1)
        fJ2 = NonLinearF(in_dim=3, baseinit=True, hidden=3, out_dim=1)
        fJparams = list(fJ1.parameters()) + list(fJ2.parameters())
        optimizer = torch.optim.Adam(fJparams, lr=lr)
        earlystoper = EarlyStopping(patience=100, min_delta=0.01, verbose=False)

        loss_func = nn.MSELoss()
        log = list()
        doData = deepcopy(self.trainData)

        for itera in range(num_iters + 1):
            # do(M1)
            doData['M1'] = fJ1(doData['Y'])
            # regenerate X2,X4,X3,X5
            doData['X2'] = self.f2(concat(doData, ['X6', 'X7', 'M1', 'Y']))
            doData['X4'] = self.f4(doData['X2'])
            doData['X3'] = self.f3(concat(doData, ['X4', 'X11']))
            doData['X5'] = self.f5(concat(doData, ['X3', 'X10', 'X11']))
            # do(M2)
            doData['M2'] = fJ2(concat(doData, ['X4', 'X6', 'X11']))

            # neg MSE loss
            predY = self.f_S_prime(concat(doData, self.S_prime + self.M))
            mse = loss_func(doData['Y'], predY)
            wnorm = regularizer(fJ1) + regularizer(fJ2)
            loss = - mse + wnorm

            if itera % (num_iters // 5) == 0:
                print('iteration: {:d}, mse: {:.3f}, wnorm: {:.3f}, lr:{:.4f}'.format(int(itera), float(mse),
                    float(wnorm),optimizer.state_dict()['param_groups'][0]['lr'])) if verbose else None

            if itera % interval == 0:
                log.append(float(mse.detach()))
                earlystoper(log[-1])
                if earlystoper.early_stop:
                    break

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.fJ1 = fJ1
        self.fJ2 = fJ2
        return log + [log[-1]] * (num_iters // interval + 1 - len(log)) # make sure all log has the same length

    def _test(self, ):
        log = list()
        for Data in self.testDatas:
            Y = Data['Y']
            Y_pred = self.f_S_prime(concat(Data, self.S_prime + self.M))

            mse = torch.mean((Y - Y_pred) ** 2).item()
            log.append(mse)

        return log

    def estimate(self, S_prime):
        for node in self.M:
            assert node not in S_prime, 'Mistake: S_prime must not contain any X_M'
        self.S_prime = S_prime
        print('Train_split: {}, S_prime: {}'.format(self.trainsplit, ','.join(self.S_prime)))

        fSlog = self._f_S_prime()
        hlog = self._hstar()

        testlog = self._test()

        return fSlog, hlog, testlog