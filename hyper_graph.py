from hypernetx import Hypergraph, drawing
from networkx import fruchterman_reingold_layout as layout
import matplotlib.pyplot as plt
import ot
import numpy as np
from pprint import pprint
Basis = tuple[int,int]
class HyperGraph():
    # bases are pairs of node and edge (node, edge)
    def __init__(self, edges:list[list[int]]):
        self.g =  Hypergraph(edges)
        self.edges = edges
        self.bases = [(node, edge) for edge, nodes in enumerate(edges) for node in nodes ]
        
    def distance(self, basis_from: Basis, basis_to: Basis):
        if basis_from == basis_to:
            return 0
        adj_edges = [i for i in range(len(self.edges)) if basis_from[0] in self.edges[i]]
        return min([self.g.edge_distance(str(edge), self.g.edges[str(basis_to[1])]) for edge in adj_edges]) + 1  # type: ignore

    def get_distr(self, basis: Basis):
        (node, _) = basis; 
        coined_nbrs = [basis] + list(filter(lambda x: x[0]==node and x!=basis, self.bases))      
        deg_n = len(coined_nbrs)
        coined_distr = [2 / deg_n  - 1] + [2 / (deg_n)] * (deg_n - 1)
        nbrs = []; distr = []
        for i in range(len(coined_nbrs)):
            shifted_nbrs = [coined_nbrs[i]] + list(filter(lambda x: x[1]==coined_nbrs[i][1] and x!=coined_nbrs[i], self.bases))      
            nbrs += shifted_nbrs
            deg_e = len(shifted_nbrs)
            distr += list(coined_distr[i] * np.array([2 / deg_e  - 1] + [2 / (deg_e)] * (deg_e - 1)))
        return distr, nbrs

    def get_distr_classic(self, basis:Basis):
        (node, _) = basis; 
        coined_nbrs = [basis] + list(filter(lambda x: x[0]==node and x!=basis, self.bases))      
        deg_n = len(coined_nbrs)
        coined_distr = [1 / deg_n] * deg_n
        nbrs = []; distr = []
        for i in range(len(coined_nbrs)):
            shifted_nbrs = [coined_nbrs[i]] + list(filter(lambda x: x[1]==coined_nbrs[i][1] and x!=coined_nbrs[i], self.bases))      
            nbrs += shifted_nbrs
            deg_e = len(shifted_nbrs)
            distr += list(coined_distr[i] * np.array([1 / deg_e]*deg_e))
        return distr, nbrs

    def signed_measure(self, basis_from: Basis, basis_to: Basis):
        x, source_nbr = self.get_distr(basis_from)
        y, target_nbr = self.get_distr(basis_to)
        for i in range(len(source_nbr)-1,-1,-1):    
                if x[i] < 0:
                    target_nbr.append(source_nbr.pop(i))
                    y.append(-x.pop(i))
        for i in range(len(target_nbr)-1,-1,-1):    
                if y[i] < 0:
                    source_nbr.append(target_nbr.pop(i))
                    x.append(-y.pop(i))
        return (x, source_nbr), (y, target_nbr)       

    def ricci(self, basis_from: Basis, basis_to: Basis):
        self.basis_check(basis_from)
        self.basis_check(basis_to)
        output={}
        (x, source_nbr), (y, target_nbr) = self.signed_measure(basis_from, basis_to)
        d=[[self.distance(snbr, tnbr) for tnbr in target_nbr] for snbr in source_nbr]
        key = (str(basis_from[0])+", "+str(self.edges[basis_from[1]]) + " -> "+ str(basis_to[0]) + ", "+ str(self.edges[basis_to[1]]))
        output[key]= 1 - ot.emd2(np.array(x), np.array(y), d) / self.distance(basis_from,basis_to)   # type: ignore
        return output

    def basis_check(self, basis: Basis):
        try:
            self.bases.index(basis)
        except ValueError:
            raise ValueError('Invalid basis')        

    def list_edges(self, node: int):
        print("Edges of node", node, ":", ["id: " + str(basis[1]) + ", " + str(self.edges[basis[1]]) for basis in self.bases if basis[0] == node])

    def plot(self):
        drawing.draw(self.g)

def comp_graph():
    node = list(range(0, 4))
    return [[a, b] for idx, a in enumerate(node) for b in node[idx + 1 :]]

def bin_tree(depth: int):
    edges = [[0, 1], [0, 2]]
    for d in range(depth):
        for i in range(2 ** d - 1, 2 ** (d + 1) - 1):
            edges += [[i, 2 * i + 1], [i, 2 * i + 2]]
    return edges

def line(length: int):
    edges = []
    for i in range(length - 1):
        edges.append([i, i + 1])
    return edges

print(line(4))
H = HyperGraph(comp_graph())

H.list_edges(0)
pprint(H.ricci((1, 0),(0, 0)))
pprint(H.ricci((1, 0),(2, 3)))

H = HyperGraph(bin_tree(5))
H.list_edges(3)
H.list_edges(7)
pprint(H.ricci((3, 8),(7, 16)))

H = HyperGraph(line(10))
H.list_edges(3)
H.list_edges(4)
pprint(H.ricci((3, 2),(4, 3)))



    
