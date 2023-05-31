import os
import numpy as np
import multiprocessing as mp
from findoptset import Find_OptSets

import sys
sys.path.append('./utils/')
from merge import merge
from jsonio import save


if __name__ == '__main__':
    
    # 45 random partition of train/test envs
    BASE = '../../data/impc'
    trainsplits = list()
    for filename in os.listdir(BASE):
        if 'csv' not in filename and 'ipynb_checkpoints' not in filename:
            trainsplits.append(filename)
            
    with mp.Pool(9) as pool:
        recorders = pool.map(Find_OptSets, trainsplits)
        pool.close()
        pool.join()
        
    recorder = merge(recorders)
    save(recorder,'record.json')