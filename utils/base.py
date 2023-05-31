from tensor import dfTotensor
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class BaseSet():
    def __init__(self,trainsplit,path,parser,need_norm):
        self.trainsplit = trainsplit
        self.parser = parser
        self.path = path
        
        self.S = self.parser['X_S']
        self.M = self.parser['X_M']
        self.doStar = self.parser['X_dostar']

        # read, normalize, and convert to torch.tensor
        trainfilename = os.path.join(path, '{}.csv'.format(trainsplit))
        self.trainDF = pd.read_csv(trainfilename)

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