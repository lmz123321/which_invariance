# debugging: /code_test/eqcls_search/3_py_dag2mag.ipynb
import numpy as np
from graphical_models import DAG
from ancestral_graph import AncestralGraph
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# import r library and function
robjects.packages.importr('ggm')
_ = robjects.r('''source('dag2mag.r')''')
rdag2mag = robjects.globalenv['DAG.to.MAG']


def get_rows(rmat):
    '''
    given a r-matrix, return row names (list of string)
    '''
    return list(robjects.r['rownames'](rmat))

def mat2mag(rmagmat):
    '''
    convert r-mag-adjmat to MAG object in py-graphical_models
    0->1; 10-10; 100<->100
    '''
    nodes = get_rows(rmagmat)
    rmagmat = np.array(rmagmat)
    dis = set(); bidis = set(); undis = set()

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if rmagmat[i,j]==0 and rmagmat[j,i]==1:
                dis.add((nodes[j],nodes[i]))
            if rmagmat[i,j]==1 and rmagmat[j,i]==0:
                dis.add((nodes[i],nodes[j]))
            if rmagmat[i,j]==10 and rmagmat[j,i]==10:
                undis.add((nodes[i],nodes[j]))
            if rmagmat[i,j]==100 and rmagmat[j,i]==100:
                bidis.add((nodes[i],nodes[j]))
    
    mag = AncestralGraph(nodes=nodes,directed=dis,bidirected=bidis,undirected=undis)
    return mag

def dag2mag(dag,latent,selection):
    '''
    DAG->MAG given latent and selection sets (a py-wrapper of BillShipley/CauseAndCorrelation/DAG.to.MAG.R)
    - dag,mag: both objects of py-graphical_models 
    - latent,selection: set of string
    - naming rule: Y,X1,...,Xd-1; all r functions/objects rXXX
    '''
    # args checkings
    d = len(dag.nodes)
    assert dag.nodes=={'Y'}.union({'X{}'.format(ind+1) for ind in range(d-1)}),'DAG node names should be Y,X1,...,Xd-1.'
    assert 'Y' not in latent and 'Y' not in selection,'Y should not be in lat./sel. set.'
    
    # dag->rdag (in r-ggm format)
    rdag = list()
    for node in dag.nodes:
        pa = dag.parents_of(node)
        if len(pa)>0:
            strpa = '+'.join(pa)
            formulae = '{}~{}'.format(node,strpa)
            rdag.append(robjects.Formula(formulae))
    rdagmat = robjects.r['DAG'](*rdag)
    
    # lat./sel. in r format
    rlatent = robjects.StrVector(latent.union(selection))
    rselection = robjects.StrVector(selection) if selection!=set() else robjects.r('NULL')
    
    # call the r-dag2mag function 
    rmagmat = rdag2mag(rdagmat,rlatent,rselection)
    
    # post-processing
    if np.array(rmagmat).shape[0] == 1:
        # mag with 1 node (Y)
        mag = AncestralGraph(nodes=dag.nodes.difference(latent.union(selection)))
    else:
        # mag with >=2 nodes
        mag = mat2mag(rmagmat)
    
    return mag
