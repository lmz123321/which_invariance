from copy import deepcopy
import numpy as np

def sort_topology(llist,order):
    '''
    llist is a list of string, e.g., ['X1','X3']
    sort it with topology order, for example, order = ['Y','X3','X1','X2'], then, return ['X3','X1']
    '''
    for node in llist:
        assert node in order,'Node {} not in the given topology order {}.'.format(node,order)
    
    _list = deepcopy(llist)
    _list.sort(key=lambda node:np.where(np.array(order)==node)[0].item())
    return _list