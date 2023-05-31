import sys
sys.path.append('./utils/')
from jsonio import load,save
from merge import merge

import os
from findoptset import train,OptSet
import multiprocessing as mp
from functools import partial
from time import time
import argparse



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimal subset selection.')
    parser.add_argument('--parsepath','-p', type=str,default='./cache/graphparse.json',
                        help='path to the parse graph json file.')
    parser.add_argument('--eqsetpath','-e', type=str,default='./cache/eqsubsets.json',
                        help='path to equivalence subsets.')
    parser.add_argument('--dataroot','-d', type=str,default='./data/simulation/',
                        help='root of your data.')
    args = parser.parse_args()
    
    parsepath = args.parsepath
    eqsetpath = args.eqsetpath
    dataroot = args.dataroot
    
    # whether the whole stable set is optimal
    parser = load(parsepath)
    eqsets = load(eqsetpath)
    
    S_prime_all = parser['X_S'] if parser['IsFullSet'] else eqsets
    print('Is fullset: {}, search over {} subsets.'.format(parser['IsFullSet'],len(S_prime_all)))
    splits = [s for s in os.listdir(dataroot) if not s.endswith('.csv')]
    
    train_par = partial(train,OptSet=OptSet,S_prime_all=S_prime_all,parser=parser,path=dataroot,need_norm=True)
    start = time()
    with mp.Pool(3) as pool:
        records = pool.map(train_par, splits)
        pool.close()
        pool.join()
    end = time()
    print('Time cost: {:.3f} h'.format((end-start)/3600))
    
    record = merge(records)
    save(record,'record.json')
    
        