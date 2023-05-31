import os
import argparse
import networkx as nx
from dag2mag import dag2mag
from graphical_models import DAG
from util import toset,tolist,powerset,neighbors,inplace_union,print_eqclses

import sys
sys.path.append('./utils/')
from jsonio import load,save

def _recover(dag,lat,sel):
    '''
    recover the eqcls given a latent/selection set
    note: an eqcls is a (frozen)set of (frozen)sets, e.g., {{1,2},{1}}; (because python does not support a set of sets)
          the set of all eqclses is hence a set of (frozen)sets of (frozen)sets
          set()==frozenset() = True
    '''
    mag = dag2mag(dag,lat,sel)
    nodes = mag.nodes.difference('Y')
    neigy = neighbors(mag)
    
    if neigy == set():
        return set([powerset(nodes)]) # a set with a single element
    else:
        P = set()
        for _sel in powerset(neigy):
            _lat = neigy.difference(_sel)
            _P = _recover(dag,lat.union(_lat),sel.union(_sel))
            _P = inplace_union(_P,_sel)
            P = P.union(_P)
        return P
    
def recover(dag):
    '''
    wrapper of equivalence classes searching
    example: 
        # dag = DAG(arcs={('Y','X1'),('X1','X2'),('Y','X3'),('X3','X4'),('X1','X3')})
        # eqclses = recover(dag)
        # print_eqclses(eqclses)
    '''
    neigy = dag.neighbors_of('Y')
    nodes = dag.nodes.difference('Y')
    
    if neigy == set():
        return set([powerset(nodes)])
    else:
        P = set()
        for _sel in powerset(neigy):
            _lat = neigy.difference(_sel)
            _P = _recover(dag,_lat,_sel)
            _P = inplace_union(_P,_sel)
            P = P.union(_P)
        return P


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='G-equivalence classes searching.')
    parser.add_argument('--file','-f', type=str, help='path to the stable DAG (gml) file.')
    parser.add_argument('--save','-s', type=str, default='./cache/',help='path to save the output.')
    
    args = parser.parse_args()
    assert args.file.endswith('gml'),'Expect input in networkx gml file.'
    
    # gml -> graphical_models.DAG
    nxg = nx.read_gml(args.file)
    dag = DAG(arcs=nxg.edges)
    
    eqclses = recover(dag)
    eqclses = tolist(eqclses)
    
    os.makedirs(args.save,exist_ok=True)
    save(eqclses,os.path.join(args.save,'eqclses.json'))
    
    # subsets need to search
    subsets = list()
    for eqcls in eqclses:
        # find shortest list in a list of lists
        subset = min(eqcls,key=len)
        subset.sort()
        subsets.append(subset)
    save(subsets,os.path.join(args.save,'eqsubsets.json'))
    
    
    
