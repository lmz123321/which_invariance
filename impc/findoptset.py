import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np

import sys
sys.path.append('./utils/')
from tensor import dfTotensor

class NonLinearF(nn.Module):
    def __init__(self, in_dim, baseinit, out_dim=1, hidden=5):
        super(NonLinearF, self).__init__()

        self.in_dim = in_dim
        modules = [nn.Linear(in_dim, hidden),nn.Sigmoid(),nn.Linear(hidden, out_dim)]

        self.function = nn.Sequential(*modules)
        self.trained = False

        if baseinit:
            for m in self.function:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.function(x)
    
def do_train(target, covariates, function, optimizer, loss_func, num_iters=50000):
    """
    do the training, predict target using covariates
    target and covariates are respectively with shape (B,1) (B,n)
    """
    for itera in range(num_iters + 1):
        prediction = function(covariates)
        loss = loss_func(prediction, target)
        #if itera % (num_iters // 2) == 0:
        #    print('iteration: {:d}, loss: {:.7f}'.format(int(itera), float(loss)))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    function.trained = True
    return function

class OptNodeSets():
    """
    given a training split, e.g. 02471012, first read from BASE/02471012.csv
    then, for a given S' in S_all
    - estimate fs':
        learn st-eq (x2=f2(x1,x3,x5,y), x4=f4(x2)), shuffle Xdo*={X_4,X_5} and regenerate its descendant, train Y=fs'(Xs',do(Xdo*))
    - estimate h*:
        replace X5 by J(PA5), regenerate its descendant, and compute worse-case risk
    """

    def __init__(self, trainsplit, seed=1234, need_norm=False):
        BASE = '../../data/impc/'
        trainfilename = os.path.join(BASE, '{}.csv'.format(trainsplit))
        self.trainDF = pd.read_csv(trainfilename)

        self.need_norm = need_norm
        if self.need_norm:
            for var in ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'Y']:
                mean = self.trainDF[[var]].mean().values[0]
                std = self.trainDF[[var]].std().values[0]
                self.trainDF[[var]] = (self.trainDF[[var]] - mean) / std

        self.testfolder = os.path.join(BASE, '{}'.format(trainsplit))

        self.f_regen = NonLinearF(in_dim=4, baseinit=True)
        self.f_regen4 = NonLinearF(in_dim=1, baseinit=True)

        self.seed = seed
        self.trainsplit = trainsplit

        self.estimate_f_regen()
        self.estimate_f_regen4()

    def estimate_f_regen(self, ):
        """
        learn x_2 = self.f_regen(x_1,x_3,x_5,y)
        """
        X = dfTotensor(self.trainDF[['X_1', 'X_3', 'X_5', 'Y']])
        Y = dfTotensor(self.trainDF[['X_2']])

        optimizer = torch.optim.SGD(self.f_regen.parameters(), lr=0.01)
        loss_func = nn.MSELoss()

        self.f_regen = do_train(Y, X, self.f_regen, optimizer, loss_func, num_iters=5000)

        for param in self.f_regen.function.parameters():
            param.requires_grad = False

    def estimate_f_regen4(self, ):
        """
        learn x_4 = self.f_regen(x_2)
        """
        X = dfTotensor(self.trainDF[['X_2']])
        Y = dfTotensor(self.trainDF[['X_4']])

        optimizer = torch.optim.SGD(self.f_regen4.parameters(), lr=0.01)
        loss_func = nn.MSELoss()
        self.f_regen4 = do_train(Y, X, self.f_regen4, optimizer, loss_func, num_iters=5000)

        for param in self.f_regen4.function.parameters():
            param.requires_grad = False

    def estimate_f_S_prime(self, ):
        """
        sample from p* by shffle X_do*={X_4,X_5} and regenerate X_i by f
        train Y = f_S'(X_S',do(X_5)) in p*
        """
        
        # shuffle X_do*={X_4,X_5}
        shufTrainDF = self.trainDF.copy()
        shufTrainDF.loc[:, 'X_4'] = shufTrainDF.loc[:, 'X_4'].sample(frac=1, random_state=self.seed).values
        shufTrainDF.loc[:, 'X_5'] = shufTrainDF.loc[:, 'X_5'].sample(frac=1, random_state=self.seed).values

        # regenerate X_2 by the trained f_regen
        X = dfTotensor(shufTrainDF.loc[:, ['X_1', 'X_3', 'X_5', 'Y']])
        pred = self.f_regen(X).detach().numpy()

        shufTrainDF.loc[:, 'X_2'] = pred

        # train the y = f_S'(x_S',do X_5)
        XX = dfTotensor(shufTrainDF[self.S_prime + ['X_5']])
        YY = dfTotensor(shufTrainDF[['Y']])

        optimizer = torch.optim.SGD(self.f_S_prime.parameters(), lr=0.01)
        loss_func = nn.MSELoss()

        self.f_S_prime = do_train(YY, XX, self.f_S_prime, optimizer, loss_func, num_iters=10000)

        for param in self.f_S_prime.function.parameters():
            param.requires_grad = False

    def estimate_hstar_S_prime(self, num_iters=10000):
        """
        estimate h*(S')
            1. generate samples from P(J_theta)
                 - replace X_5 by J_theta(PA_5)
                 - regenerate X_i by f(PA_i) for X_i is a descent of X_5
            2. calculate Y_hat = f_S'(X_S',do(X_5)), where X_S',X_5~P(J_theta)
            3. compute negMSELoss -||Y-Y_hat|| and optimize \theta
        """
        Y = dfTotensor(self.trainDF[['Y']])
        X_1 = dfTotensor(self.trainDF[['X_1']])
        X_3 = dfTotensor(self.trainDF[['X_3']])

        optimizer = torch.optim.SGD(self.f_J_theta.parameters(), lr=0.05)
        loss_func = nn.MSELoss()
        loss_log = list()

        for itera in range(num_iters + 1):
            # replace X_5 by J_theta(Y)
            pred_X_5 = self.f_J_theta(Y)
            # regenerate X_2 = f_regen(pred_X_5,X_1,X_3,Y)
            pred_X_2 = self.f_regen(torch.cat([X_1, X_3, pred_X_5, Y], dim=1))

            # regenerate X_4 = f_regen4(pred_X_2)
            pred_X_4 = self.f_regen4(pred_X_2)

            # calculate pred_Y = f_S_prime(X_S', do(X_5))
            X_S_all = {'X_1': X_1, 'X_2': pred_X_2, 'X_3': X_3, 'X_4': pred_X_4}
            X_S_prime_X_M = list()
            for S in self.S_prime:
                X_S_prime_X_M.append(X_S_all[S])
            X_S_prime_X_M.append(pred_X_5)

            # prediction
            pred_Y = self.f_S_prime(torch.cat(X_S_prime_X_M, dim=1))

            # we want to maximize this loss, so add a negative
            loss = - loss_func(Y, pred_Y)
            if itera % (num_iters // 10) == 0:
                #print('iteration: {:d}, loss: {:.7f}'.format(int(itera), - float(loss)))
                loss_log.append(- float(loss.detach()))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.f_J_theta.trained = True
        for param in self.f_J_theta.function.parameters():
            param.requires_grad = False

        return loss_log

    def estimate(self, S_prime):
        
        print('Train_split: {}, S_prime: {}'.format(self.trainsplit, ','.join(S_prime)))
        self.f_S_prime = NonLinearF(in_dim=len(S_prime) + 1, baseinit=True)
        self.f_J_theta = NonLinearF(in_dim=1, baseinit=True)
        self.S_prime = S_prime

        self.estimate_f_S_prime()
        negMSELosses = self.estimate_hstar_S_prime()

        return negMSELosses

    def test(self, S_prime):
        # test
        error_log = list()

        for filename in os.listdir(self.testfolder):
            testDF = pd.read_csv(os.path.join(self.testfolder, filename))

            if self.need_norm:
                # normalization
                for var in ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'Y']:
                    mean = testDF[[var]].mean().values[0]
                    std = testDF[[var]].std().values[0]
                    testDF[[var]] = (testDF[[var]] - mean) / std

            X_test = dfTotensor(testDF[self.S_prime + ['X_5']])
            Y_test = dfTotensor(testDF[['Y']])

            Y_pred = self.f_S_prime(X_test)
            mse = torch.mean((Y_test - Y_pred) ** 2).item()

            error_log.append(mse)

        return error_log
    
def Find_OptSets(trainsplit):
    """
    - given a trainsplit (which is a str)
    - create a OptNodeSets object and search for the h* of every S_prime in this trainsplit
    - this function return a recorder, which is a dict, recording h* and maxTestErrors
    """

    S_prime_all = [[],['X_1'], ['X_2'], ['X_3'], ['X_4'],
                   ['X_1', 'X_2'], ['X_1', 'X_4'], ['X_1', 'X_3'], ['X_2', 'X_3'], ['X_2', 'X_4'], ['X_3', 'X_4'],
                   ['X_1', 'X_2', 'X_3'], ['X_1', 'X_2', 'X_4'], ['X_1', 'X_3', 'X_4'], ['X_2', 'X_3', 'X_4'],
                   ['X_1', 'X_2', 'X_3', 'X_4']]

    # we only need to compute S_prime = ['X_1'] and ['X_3'] in 4.21
    # remove this line if you want to use S_all
    #S_prime_all = [[],['X_1']]

    # recorder has two level of keys
    # S_prime, and h_stars/test_errors
    # different trainsplit results appended under the same S_prime

    recorder = dict()
    for S_prime in S_prime_all:
        save_name = ','.join(S_prime) if len(S_prime)!=0 else 'empty'
        recorder[save_name] = dict()
        recorder[save_name]['h_stars'] = list()
        recorder[save_name]['test_errors'] = list()
    
    optnodeset = OptNodeSets(trainsplit=trainsplit, need_norm=True)

    for S_prime in S_prime_all:
        save_name = ','.join(S_prime) if len(S_prime)!=0 else 'empty'
        h_star = optnodeset.estimate(S_prime=S_prime)
        recorder[save_name]['h_stars'].append(h_star)
        test_error = optnodeset.test(S_prime=S_prime)
        recorder[save_name]['test_errors'].append(test_error)

    return recorder