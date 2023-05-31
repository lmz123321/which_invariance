import os
from findoptset import OptSet
from itertools import combinations
import multiprocessing as mp
from functools import partial
from time import time

import sys
sys.path.append('./utils/')
from neuralnet import train
from merge import merge
from jsonio import save

if __name__ == '__main__':
    
    path = './data/simulation/'
    splits = [s for s in os.listdir(path) if not s.endswith('.csv')]
    
    # note that the node naming rule in the code is a bit different from Fig.4a in the paper
    # we use code to transform
    stableNodes = ['X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X10', 'X11']
    power_set = [[]]
    for n in range(len(stableNodes)):
        for sett in list(combinations(stableNodes, n + 1)):
            power_set.append(list(sett))
            
    #S_prime_all = power_set[:4]
    S_prime_all = power_set
    train_par = partial(train, OptSet=OptSet, S_prime_all=S_prime_all, path=path)
    
    start = time()
    with mp.Pool(3) as pool:
        records = pool.map(train_par, splits)
        pool.close()
        pool.join()
    end = time()
    print('Time cost: {:.3f} h'.format((end-start)/3600))
    
    record = merge(records)
    save(record,'record.json')

