# Select the needed variables (i.e. B1-B22,C1-C3,S4,E) from aggrADNI30T15TAge25Nodes
# Rename them into X1-X25,Y,E
# Generate data for the given trainsplits and corresponding testsplits

import pandas as pd
import os
import numpy as np
from itertools import combinations


class ADNISelect():
    def __init__(self, need_sim=False):
        super(ADNISelect, self).__init__()

        self.covariates = ['B{}'.format(ind + 1) for ind in range(22)] + ['C1', 'C2', 'C3', 'S4', 'E']

        self.outputnames = ['X{}'.format(ind + 1) for ind in range(25)] + ['Y', 'E']

        self.covToout = dict(zip(self.covariates, self.outputnames))

        ages = [ind for ind in range(100)]
        indices = list()
        for age in ages:
            if age < 60:
                indices.append(1)
            elif age < 65:
                indices.append(2)
            elif age < 70:
                indices.append(3)
            elif age < 75:
                indices.append(4)
            elif age < 80:
                indices.append(5)
            elif age < 85:
                indices.append(6)
            else:
                indices.append(7)
        self.ageToindex = dict(zip(ages, indices))

        if need_sim:
            self.aggrDF = pd.read_csv('aggrADNI30T15TAge25NodesSim.csv')
        else:
            self.aggrDF = pd.read_csv('aggrADNI30T15TAge25Nodes.csv')

        self.selectDF = self.aggrDF[self.covariates].copy(deep=True)
        # map ScanY into indices
        self.selectDF['E'] = self.selectDF['E'].map(self.ageToindex)

    def generate(self, envids):
        r"""
        Functs: - choose envids from self.selectDF
                - re-order their envid from 1-max
                - write the results under filefolder with name ADNIB2S_XXX.csv
        """
        assert set(envids) <= set((1, 2, 3, 4, 5, 6, 7)), 'Expect envid choosen from (1,2,3,4,5,6,7)'
        # choose envids
        outputDF = self.selectDF.loc[self.selectDF['E'].isin(envids)].copy(deep=True)
        # reorder E from 1 to max
        indexToindex = dict(zip(list(envids), [ind + 1 for ind in range(len(envids))]))
        outputDF['E'] = outputDF['E'].map(indexToindex)

        # rename B1-B22,C1-C3,S4,E into X1-X25,Y,E
        outputDF = outputDF.rename(columns=adniselect.covToout).set_index('X1')

        return outputDF


if __name__ == '__main__':
    BASE = '/home/liumingzhou/data/CausallyInvariant_output/Lasso/ADNI20sNodes/FindOptSets'
    os.makedirs(BASE, exist_ok=True)

    #envids = [(1, 4, 5), (2, 3, 5), (2, 4, 5), (2, 5, 7), (4, 5, 6),
    #          (1, 2, 3, 5), (1, 2, 4, 5), (1, 4, 5, 6), (2, 3, 4, 5), (2, 3, 5, 6), (2, 3, 5, 7), (2, 4, 5, 6),
    #          (2, 4, 5, 7), (3, 4, 5, 6), (3, 4, 5, 7), (3, 4, 6, 7), (4, 5, 6, 7)]

    envids = list(combinations((1,2,3,4,5,6,7),3)) + list(combinations((1,2,3,4,5,6,7),4)) + [(1,2,3,4,5,6,7)]

    adniselect = ADNISelect(need_sim=False)

    for envid in envids:
        # the train split
        trainDF = adniselect.generate(envids=envid)
        trainDF.to_csv(os.path.join(BASE, '{}.csv'.format(''.join(str(e) for e in envid))))

        os.makedirs(os.path.join(BASE, '{}'.format(''.join(str(e) for e in envid))), exist_ok=True)
        # the test splits
        testids = [ind + 1 for ind in range(7) if ind + 1 not in envid]
        for testid in testids:
            testDF = adniselect.generate(envids=[testid])
            testDF.to_csv(os.path.join(BASE, '{}'.format(''.join(str(e) for e in envid)), '{}.csv'.format(testid)))