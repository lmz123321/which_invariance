import argparse
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.utils.cit import kci,fisherz
from causallearn.graph.GraphNode import GraphNode
import os
import pandas as pd
import numpy
import networkx as nx
from graphparser import ComponentFinder

def order(string,maxnum=2000):
    '''
    sort order, for X1 return 1, for Y return 200 (so that Y is always sorted to the last)
    '''
    if 'X' in string:
        return int(string.replace('X',''))
    elif string=='Y':
        return maxnum
    else:
        raise ValueError('Except input to be X.. or Y.')

def sort_alphabet(parser):
    '''
    for a graphparser.record
    convert sets to lists with sorted order (X1,X2,...,)
    '''
    nochange = ['Name','topology','IsFullSet']
    for key in parser.keys():
        if key in nochange:
            continue
        if type(parser[key])== set:
            parser[key] = list(parser[key])
            parser[key].sort(key=order)
        else:
            for subkey in parser[key].keys():
                parser[key][subkey] = list(parser[key][subkey])
                parser[key][subkey].sort(key=order)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hetergenous causal discovery and component detection.')
    parser.add_argument('--file','-f', type=str, default='134581011151617.csv',
                        help='filename of the data (csv) file.')
    parser.add_argument('--path','-p', type=str, default='./data/simulation',
                        help='path to base of the data files.')
    parser.add_argument('--cache','-c', type=str, default='./cache/',
                        help='where to save cache files.')
    args = parser.parse_args()
    
    file = args.file; path = args.path; cache = args.cache
    filename = os.path.join(path,file)
    
    # CD-NOD (one may add Background knowledges, see documentation of the causal-learn API for details)
    df = pd.read_csv(filename)
    data = df.values
    
    citest_cache_file = os.path.join(cache,'cit_cache.json')
    aug = cdnod(data=data[:, :-1], c_indx=data[:, -1][:, None],indep_test=kci, 
        background_knowledge=None,show_progress=False,cache_path=citest_cache_file)
    
    aug.to_nx_graph()
    nxaug = aug.nx_graph
    
    # renaming: from 0,1,... to Y,E,X1,...
    varnames = df.columns.tolist()
    labels = {i:varnames[i] for i in range(len(varnames))}
    nxaug = nx.relabel_nodes(nxaug,labels)
    
    # debugging: use the gt-aug graph 
    if file=='134581011151617.csv':
        gtaug = nx.read_gml('./data/gtaug.gml')
        aug = gtaug
    else:
        aug = nxaug
    nx.write_gml(aug,os.path.join(cache,'auggraph.gml'))
        
        
    # local component detection
    comfinder = ComponentFinder(aug,name=file)
    comfinder.detect()
    assert 'Y' not in comfinder.recorder['X_M'],'Y is a mutable variable, which violates our assumption.'
    
    stable = aug.copy()
    stable.remove_nodes_from(comfinder.recorder['X_M'].union({'E'}))
    nx.write_gml(stable,os.path.join(cache,'stablegraph.gml'))

    sort_alphabet(comfinder.recorder)
    comfinder.to_json(filename=os.path.join(cache,'graphparse.json'))
    
    