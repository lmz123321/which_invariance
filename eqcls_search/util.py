from itertools import chain, combinations

def toset(listoftup):
    '''
    convert a list of tuples to that of sets
    '''
    return [frozenset(element) for element in listoftup]

def tolist(eqclses):
    '''
    convert an eqclses (a set of sets of sets) to a jsonable object (a list of lists of lists)
    '''
    copys = list()
    for eqcls in eqclses:
        copy = list()
        for baseset in eqcls:
            copy.append(list(baseset))
        copys.append(copy)
    return copys

def powerset(iterable):
    '''
    generate the powerset of a iterable object
    data: a (frozen)set of (frozen)sets
    '''
    return frozenset(chain.from_iterable(toset(combinations(iterable, r)) 
                        for r in range(len(iterable)+1)))

def neighbors(mag,node='Y'):
    '''
    return neig(Y), a set of nodes that are connected to Y
    note: X - Y  mag.neighbors_of
          X<->Y  mag.spouses_of
    '''
    neigs = set()
    neigs = neigs.union(mag.parents_of(node))     
    neigs = neigs.union(mag.children_of(node))
    neigs = neigs.union(mag.neighbors_of(node))
    neigs = neigs.union(mag.spouses_of(node))
    
    return neigs

def inplace_union(eqclses,sel):
    '''
    eqclses is a set of sets of base-sets, sel is a set
    we hope to union the sel into every base-set in the eqclses
    we do this by creating a new copy of eqclses
    '''
    copys = set()
    for eqcls in eqclses:
        copy = set()
        for baseset in eqcls:
            copy.add(baseset.union(sel))
        copy = frozenset(copy)
        copys.add(copy)
    return copys

def print_eqclses(eqclses, verbose=True):
    '''
    custom defined print of an eqclses
    it will also check the validity of detected eqclses
    '''
    num_subsets = 0
    num_nodes = 0
    for ind,eqcls in enumerate(tolist(eqclses)):
        print('{}-th eqcls:'.format(ind+1),eqcls) if verbose else None
        num_subsets += len(eqcls)
        for subset in eqcls:
            if len(subset)>num_nodes:
                num_nodes = len(subset)
    assert num_subsets == 2**num_nodes, 'Got {} subsets, while having {} (stable) nodes.'.format(num_subsets,num_nodes)
