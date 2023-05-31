import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensor import dfTotensor

def regularizer(model, lamb=0.01, p=2, items=['weight']):
    loss = 0
    for name, value in model.named_parameters():
        for item in items:
            if name.endswith(item):
                loss += torch.norm(value,p)
    return lamb*loss
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience, min_delta, verbose=False):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif abs(self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif abs(self.best_loss - val_loss) < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}") if self.verbose else None
            if self.counter >= self.patience:
                #print('INFO: Early stopping')
                self.early_stop = True


def _weightcut(module,threshold=0.05,eps=1e-5):
    for name,param in module.named_parameters():
        param[abs(param)<=threshold] = eps
    

class NonLinearF(nn.Module):
    r"""
    Functs: - use .fit to train the model,
            - use .forward (for torch.tensor) or .predict (for pd.DataFrame) to do inference (or training of f_J_theta)
    """

    def __init__(self, in_dim, baseinit, hidden=1, out_dim=1, nonlin=True, patience=1000, min_delta=0.01,
                 verbose=False):
        super(NonLinearF, self).__init__()

        self.in_dim = in_dim
        modules = [nn.Linear(in_dim, hidden), nn.Sigmoid(), nn.Linear(hidden, out_dim)] if nonlin else [
            nn.Linear(in_dim, out_dim)]

        self.function = nn.Sequential(*modules)
        self.trained = False
        self.verbose = verbose
        self.earlystoper = EarlyStopping(patience=patience, min_delta=min_delta, verbose=False)

        if baseinit:
            for m in self.function:
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def fit(self, covariates, target, opt, num_iters, lr, earlystop=True, weightcut=False,step=None, gamma=None, log=False):

        if isinstance(covariates, pd.core.frame.DataFrame):
            covariates = dfTotensor(covariates)
            target = dfTotensor(target)

        optimizer = torch.optim.SGD(self.function.parameters(), lr=lr) if opt == 'SGD' else torch.optim.Adam(
            self.function.parameters(), lr=lr)
        if step is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)

        loss_func = nn.MSELoss()
        loss_log = list()
        for itera in range(num_iters + 1):
            prediction = self.function(covariates)
            loss = loss_func(prediction, target)
            if itera % (num_iters // 5) == 0:
                if self.verbose:
                    print('iteration: {:d}, loss: {:.3f}, lr: {:.4f}'.format(int(itera), float(loss),
                                                                             optimizer.state_dict()['param_groups'][0][
                                                                                 'lr']))

            loss_log.append(float(loss.detach()))
            if earlystop:
                self.earlystoper(loss_log[-1])
                if self.earlystoper.early_stop:
                    break

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step is not None:
                scheduler.step()

        self.trained = True
        for param in self.function.parameters():
            param.requires_grad = False
            
        if weightcut:
            _weightcut(self.function)

        if log:
            return loss_log

    def forward(self, covariates):
        return self.function(covariates)

    def predict(self, covariates):
        # assert self.trained, 'NonLinearF must be trained befored prediction'
        covariates = dfTotensor(covariates)
        return self.function(covariates).detach().numpy()

def train(split, OptSet, S_prime_all, path):
    record = dict()
    for S_prime in S_prime_all:
        save_name = ','.join(S_prime) if len(S_prime) > 0 else 'empty'
        record[save_name] = dict()
        record[save_name]['hstar'] = list()
        record[save_name]['mse'] = list()

    opt = OptSet(split, path, need_norm=True)
    for S_prime in S_prime_all:
        save_name = ','.join(S_prime) if len(S_prime) > 0 else 'empty'
        fSlog, hlog, testlog = opt.estimate(S_prime)

        record[save_name]['hstar'].append(hlog)
        record[save_name]['mse'].append(testlog)

    return record