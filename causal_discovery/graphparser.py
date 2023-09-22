import networkx as nx
from copy import deepcopy
import matplotlib.pyplot as plt
import json
import os

class PDAGParser:
    def __init__(self, pdag):
        r"""
        Args: - the pdag is a networkx.DiGraph object
              - it contains undirected (<->, green) and directed (->, blue) edges
              - we assume the pdag's node's names are X1,X2,...,E,Y, so a pre-name mapping is required

              - for nomenclature, we denote 'node' as 'X1', and 'nodes' as ['X1','X2',...]
        """
        self.pdag = pdag
        if 'X1' not in list(self.pdag.nodes) or 'E' not in list(self.pdag.nodes) or 'Y' not in list(self.pdag.nodes):
            print('Warning: node names of the input PDAG are required to be Xs,E,Y ...')

        self.num_vars = len(self.pdag.nodes)
        self.nodes = ['X{}'.format(ind + 1) for ind in range(self.num_vars - 2)] + ['Y', 'E']
        self.edges = self.pdag.edges

        # TODO: add a checker, make sure 'E' point to X_M, instead of reverse
        for edge in self.edges:
            if edge[1] == 'E':
                print('Warning: the selection variable must not be pointed to.')

    def get_parents(self, node):
        r"""
        Functs: - node -> set of nodes
                - pa is the parent of node i.i.f. (pa,node) in edges and (node,pa) not in edges
        """
        assert node in self.nodes, '{} not in nodes of the PDAG'.format(node)
        parents = list()
        for pa in self.nodes:
            if (pa, node) in self.edges and (node, pa) not in self.edges:
                parents.append(pa)

        return set(parents)

    def get_children(self, node):
        r"""
        Functs: node -> a set of nodes (Ch(node))
        """
        assert node in self.nodes, '{} not in nodes of the PDAG'.format(node)
        children = list()
        for ch in self.nodes:
            if (node, ch) in self.edges and (ch, node) not in self.edges:
                children.append(ch)

        return set(children)

    def get_adjacencies(self, node):
        r"""
        Functs: node -> a set of nodes (Adj(node))
        """
        assert node in self.nodes, '{} not in nodes of the PDAG'.format(node)
        adjacencies = list()
        for adj in self.nodes:
            if (adj, node) in self.edges or (node, adj) in self.edges:
                adjacencies.append(adj)

        return set(adjacencies)

    def collect_descendants(self, node, descendants, first_level=False):
        r"""
        Functs: - a recursive search of Dec(node)
        """
        assert node in self.nodes, '{} not in nodes of the PDAG'.format(node)
        if node in descendants:
            return

        if not first_level:
            # descendant does not include the node it self
            descendants.append(node)

        children = self.get_children(node)

        if children:
            for child in children:
                self.collect_descendants(child, descendants)

    def get_descendants(self, nodes):
        r"""
        Functs: - get Dec(nodes)={Dec(node1) \cup ... \cup Dec(nodei)} in the original graph
                - note that, if node-i in nodes is descendant of node-j in nodes, then, descendants will also contain node-i
        """
        assert isinstance(nodes, set), 'Input must be a set of nodes.'
        descendants = list()

        for node in nodes:
            # a recursive implementation
            self.collect_descendants(node, descendants, first_level=True)

        return set(descendants)


class ComponentFinder:
    def __init__(self, pdag, name, check_cycles=False):
        r"""
        Functs: - take the CD-NOD outputed P-DAG as input, denote it as G
                - firstly find local components X_M, Ch(Y), X_M^0, De(X_M^0), K, IsFullSet, PA_M, X_dostar in G
                - then, create a new nx.DiGraph G_dostar, find De(X_dostar) and PA(De(X_dostar)) in it
                - finally, create a new nx.DiGraph G_doM, find De(X_M) in it
                - return a recorder dict containing all these components

        How to use:
                - pdag = pdag_remap(nx.read_gml(os.path.join(BASE,filename)))
                - comfinder = ComponentFinder(pdag, name='yourGraphName')
                - comfinder.detect()
                - comfinder.to_json(filename='graphParse.json')
        """
        self.G = PDAGParser(pdag)
        self.G_dostar = None
        self.G_doM = None

        self.recorder = dict()
        self.recorder['Name'] = name
        topology = list(nx.topological_sort(pdag))
        self.recorder['topology'] = topology

        if check_cycles:
            self.detect_cycles()

    def detect_cycles(self, ):
        r"""
        Functs: - if Dec(node) in PA(node), then, there is a directed cycle in the input graph
                - output a warning and record this in the recorder
        """
        for node in self.G.nodes:
            parents = self.G.get_parents(node)
            descentants = self.G.get_descendants({node})
            if len(parents.intersection(descentants)) > 0:
                print('Warning, {}: {} - PA \cap De: {}'.format(self.recorder['Name'], node, parents.intersection(descentants)))

    def detect_X_M(self, ):
        r"""
        Functs: - detect mutable variables (variables that are adj/ch of 'E') and save their name in recorder
        """
        X_M = self.G.get_adjacencies(node='E')
        self.recorder['X_M'] = X_M
        self.recorder['X_S'] = set(self.G.nodes).difference(X_M).difference({'E', 'Y'})

    def detect_PA_M(self, ):
        r"""
        Functs: - detect PA(X_M), the result will be stored in a dict
                - where the key is node in X_M, and value is the corresponding parents
        """
        PA_M = dict()
        for node in self.recorder['X_M']:
            parents = self.G.get_parents(node)
            # E is not necessarily to be included
            parents.remove('E')
            PA_M[node] = parents

        self.recorder['PA_M'] = PA_M

    def detect_children_Y(self, ):
        r"""
        Functs: - detect children of Y
        """
        Ch_Y = self.G.get_children(node='Y')
        self.recorder['Ch_Y'] = Ch_Y

    def detect_X_M_0(self, ):
        r"""
        Functs: - detect X_M^0 := X_M \cap Ch(Y)
        """
        self.recorder['X_M_0'] = self.recorder['X_M'].intersection(self.recorder['Ch_Y'])

    def detect_de_X_M_0(self, ):
        r"""
        Functs: - detect De(X_M^0)
        """
        self.recorder['De_X_M_0'] = self.G.get_descendants(self.recorder['X_M_0'])

    def detect_K(self, ):
        r"""
        Functs: - detect K:=De(X_M^0)\X_M^0
        """
        self.recorder['K'] = self.recorder['De_X_M_0'].difference(self.recorder['X_M_0'])

        IsFullSet = True
        for node in self.recorder['K']:
            if node in self.recorder['Ch_Y']:
                IsFullSet = False
                break

        self.recorder['IsFullSet'] = IsFullSet

    def detect_X_dostar(self, ):
        r"""
        Functs: - detect X_dostar:=X_M^0 \cup (De(X_M^0)\Ch(Y))
        """
        self.recorder['X_dostar'] = self.recorder['X_M_0'].union(
            self.recorder['De_X_M_0'].difference(self.recorder['Ch_Y']))

    def interven_dostar(self, ):
        r"""
        Functs: - generate the graph G_dostar:=G_(do(X_dostar)), i.e., removing arrowheads into X_dostar
        """
        self.G_dostar = deepcopy(self.G)

        todel_edges = list()
        for edge in self.G_dostar.edges:
            # we can only delete directed edges into X_dostar, un-directed edges should not be deleted
            if edge[1] in self.recorder['X_dostar']:
                if tuple(reversed(edge)) not in self.G_dostar.edges:
                    todel_edges.append(edge)

        for edge in todel_edges:
            # note that, in PDAGParser, self.edges is a shallow copy of self.pdag.edges,
            # so, delete edges in self.pdag can also delete edges in self.edges
            self.G_dostar.pdag.remove_edge(edge[0], edge[1])

    def detect_de_X_dostar(self, ):
        r"""
        Functs: - detect De(X_do_star) in the intervened graph G_dostar
        """
        assert self.G_dostar is not None, 'Must intervent the graph first before detecting De(X_do_star)'

        self.recorder['De_dostar'] = self.G_dostar.get_descendants(self.recorder['X_dostar'])

    def detect_PA_de_X_dostar(self, ):
        r"""
        Functs: - detect PA(De(X_do_star)) in the intervened graph G_dostar
                - the result is stored in a dict, where the key is node in De(X_do_star), the value is its parents
        """
        PA_De_dostar = dict()
        for node in self.recorder['De_dostar']:
            # find parents in the intervened graph
            parents = self.G_dostar.get_parents(node)
            PA_De_dostar[node] = parents

        self.recorder['PA_De_dostar'] = PA_De_dostar

    def interven_M(self, ):
        r"""
        Functs: - generate the graph G_doM:=G_(do(X_M)), i.e., removing arrowheads into X_M
        """
        self.G_doM = deepcopy(self.G)

        todel_edges = list()
        for edge in self.G_doM.edges:
            # we can only delete directed edges into X_M, un-directed edges should not be deleted
            if edge[1] in self.recorder['X_M']:
                if tuple(reversed(edge)) not in self.G_dostar.edges:
                    todel_edges.append(edge)

        for edge in todel_edges:
            # note that, in PDAGParser, self.edges is a shallow copy of self.pdag.edges,
            # so, delete edges in self.pdag can also delete edges in self.edges
            self.G_doM.pdag.remove_edge(edge[0], edge[1])

    def detect_de_M_intervened(self, ):
        r"""
        Functs: - detect De(X_M) in the intervened graph G_doM
        """
        assert self.G_doM is not None, 'Must intervent the graph first before detecting De(X_M)'

        self.recorder['De_M_intervened'] = self.G_doM.get_descendants(self.recorder['X_M'])

    def detect_PA_de_M_intervened(self, ):
        r"""
        Functs: - detect PA(De(X_M)) in the intervened graph G_doM
                - the result is stored in a dict, where the key is node in De(X_M), the value is its parents
        """
        PA_de_M_intervened = dict()

        for node in self.recorder['De_M_intervened']:
            parents = self.G_doM.get_parents(node)
            PA_de_M_intervened[node] = parents

        self.recorder['PA_De_M_intervened'] = PA_de_M_intervened

    def detect(self, ):

        self.detect_X_M()
        self.detect_PA_M()
        self.detect_children_Y()
        self.detect_X_M_0()
        self.detect_de_X_M_0()
        self.detect_K()
        self.detect_X_dostar()

        self.interven_dostar()
        self.detect_de_X_dostar()
        self.detect_PA_de_X_dostar()

        self.interven_M()
        self.detect_de_M_intervened()
        self.detect_PA_de_M_intervened()

    def to_json(self, filename):

        recorder = deepcopy(self.recorder)

        for key in recorder.keys():
            if isinstance(recorder[key], set):
                recorder[key] = list(recorder[key])
            if isinstance(recorder[key], dict):
                for subkey in recorder[key].keys():
                    recorder[key][subkey] = list(recorder[key][subkey])

        with open(filename, 'w') as fout:
            json.dump(recorder, fout, indent=4)

