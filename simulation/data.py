import sys
sys.path.append('./utils/')
from steq import normal,uniform,duniform,sigmoid,identity
from jsonio import load,save
from itertools import combinations
import numpy as np
import random
import pandas as pd
import networkx as nx
import os
import argparse

class Data:
    def __init__(self,params,path):
        self.path = path
        os.makedirs(self.path,exist_ok=True)
        self.params = params
        
        self.aggrData = dict()
        for node in self.params['dag']:
            self.aggrData[node] = list()
        self.aggrData['E'] = list()
        
        # note that the node naming rule in the code is a bit different from Fig.4a in the paper
        # we use the following code to transform
        snodes = ['X2','X3','X4','X5','X6','X7','X10','X11']
        snames = ['X{}'.format(ind+1) for ind in range(8)]
        mnodes = ['M1','M2']
        mnames = ['X9','X10']
        self.XtoName = dict(zip(snodes+mnodes+['Y','E'],snames+mnames+['Y','E']))
        
            
    def _gen_env(self, e):
        """
        sample data for the environment-id e (1,2,...,num_env)
        we use the following st-eq: X_i = f_i(g_i(PA_i)) + U_i
        f_i ~ {identify,sigmoid, sinc, tanh}; g_i is linear function sampled from U(0.5,2) U(-2,-0.5)
        """
        alpha_e = e-(self.params['num_env']//2+0.5)
        aggrData = dict()
        
        for node in self.params['dag']:
            pa = self.params[node]['pa']
            # exogenous noise
            data = normal(0,0.1,size=self.params['size'])
            if len(pa)>0:
                # shape: (num_pa, size)
                pa_data = np.stack([aggrData[p] for p in pa])
                # sigmoid, sinc, or tanh
                nonlin = self.params[node]['nonlin']
                # shape (num_pa, 1)
                lin = np.array(self.params[node]['lin']).reshape(-1,1)
                # structural equation
                st_eq = functs[nonlin](np.sum(lin*pa_data,axis=0))
                data += st_eq if 'M' not in node else alpha_e*st_eq
            aggrData[node] = list(data)
        return aggrData
    
    def _csv(self,trainids):
        """
        save under /path/trainids.csv (train file) and /path/trainids/env_id.csv (test files)
        """
        
        
        testids =[e+1 for e in range(self.params['num_env']) if e+1 not in trainids]
        
        # train
        trainDF = self.aggrDF.loc[self.aggrDF['E'].isin(trainids)].copy(deep=True)
        foldername = ''.join([str(ind) for ind in trainids])
        trainDF.rename(columns=self.XtoName,inplace=True)
        trainDF.to_csv(os.path.join(self.path,'{}.csv'.format(foldername)))
        
        # test
        os.makedirs(os.path.join(self.path,foldername), exist_ok=True)
        for testid in testids:
            testDF = self.aggrDF.loc[self.aggrDF['E'].isin([testid])].copy(deep=True)
            testDF.rename(columns=self.XtoName,inplace=True)
            testDF.to_csv(os.path.join(self.path,foldername,'{}.csv'.format(testid)))
            
    def generate(self,verbose=False):
        """
        generate data for each env, aggregate them together with a domain index variable E
        then, randly assign half for training and half for testing
        """
        for e in range(self.params['num_env']):
            aggrData = self._gen_env(e+1)
            for node in self.params['dag']:
                self.aggrData[node] += aggrData[node]
            self.aggrData['E'] += [e+1]*self.params['size']
            
            if verbose:
                for node in ['M2']:
                    plt.figure()
                    plt.hist(aggrData[node])
                    plt.title('E={}, {}'.format(e+1,node))
                    
        self.aggrDF = pd.DataFrame(self.aggrData).set_index(self.params['dag'][0])
        # pick 10 out of the 20 envs as train env
        train_envs = list(combinations([e+1 for e in range(self.params['num_env'])], self.params['num_env']//2))
        # repeat for 10 times
        rand_picks = random.choices(train_envs,k=10)
        for train_env in rand_picks:
            self._csv(train_env)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulation data generation.')
    parser.add_argument('--params','-p', type=str, default='./data/simulation_params.json',
                        help='path to the stuctural equation params file.')
    parser.add_argument('--save','-s',type=str,default='./data/simulation/',
                       help='save path')
    args = parser.parse_args()

    functs = {'sigmoid':sigmoid, 'tanh':np.tanh, 'sinc':np.sinc, 'identity':identity}

    dag = nx.DiGraph()
    dag.add_edges_from([('Y','M1'),('Y','X2'),('M1','X2'),('X7','X2'),('X7','X6'),('X6','X2'),
        ('X2','X4'),('X4','X3'),('X3','X5'),('X10','X5'),('X11','X3'),('X11','X5'),
        ('X11','X10'),('X6','M2'),('X4','M2'),('X11','M2')])

    '''
    # one may regenerate random structual equation params with the following code
    nonlins = list(functs.keys())
    params = {'num_env':20, 'size':100, 'dag': list(nx.topological_sort(dag))}

    for node in dag.nodes:
        pa = list(dag.predecessors(node))
        params[node] = dict()
        params[node]['pa'] = pa
        params[node]['nonlin'] = np.random.choice(nonlins) if len(pa)>0 else None
        params[node]['lin'] = list(duniform(0.5,2,len(pa)).round(2))
    '''

    params = load(args.params)
    data = Data(params,args.save)
    data.generate()