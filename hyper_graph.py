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
        
    def distance(self, basis_from:Basis, basis_to:Basis):
        if basis_from == basis_to:
            return 0
        adj_edges = [i for i in range(len(self.edges)) if basis_from[0] in self.edges[i]]
        return min([self.g.edge_distance(str(edge), self.g.edges[str(basis_to[1])]) for edge in adj_edges]) + 1

    def get_distr(self, basis:Basis):
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

    def signed_measure(self, basis_from:Basis, basis_to:Basis):
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

    def ricci(self,basis_from:Basis, basis_to:Basis):
        output={}
        (x, source_nbr), (y, target_nbr) = self.signed_measure(basis_from, basis_to)
        d=[[self.distance(snbr, tnbr) for tnbr in target_nbr] for snbr in source_nbr]
        output[basis_from,basis_to]= 1 - ot.emd2(np.array(x), np.array(y), d) / self.distance(basis_from,basis_to) 
        return output
    
    def arc_forward(self,source,target):
        return [(edge[0],edge[1]) for edge in self.g.edges() if edge[0] == target and edge[1] !=source]

    def plot(self):
        drawing.draw(self.g)

# edges = [[0, 1], [1, 2], [2,3],[3,4]]
node = list(range(0, 4))
edges = [[a, b] for idx, a in enumerate(node) for b in node[idx + 1 :]]
H = HyperGraph(edges)
print(edges)
pprint(H.ricci((1,0),(0,0)))
pprint(H.ricci((1,0),(2,3)))
# H.plot()
    
