import matplotlib.pyplot as plt
import hypernetx as hnx
from hypernetx import Hypergraph, drawing
from networkx import fruchterman_reingold_layout as layout
from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import ot
import numpy as np
from pprint import pprint
class HyperGraph():
    def __init__(self, edges:list[list[int]]):
        self.graph =  Hypergraph(edges)
        self.basis = [(node,edge) for edge, nodes in enumerate(edges) for node in nodes ]
        
    def distance(self, basis_from:tuple[int,int], basis_to:Tuple[int,int]):
        return 0 if basis_from == basis_to else self.graph.distance(basis_from[0], basis_to[0]) + 1

    def get_distr(self, basis:tuple[int, int]):
        (node, _) = basis; 
        coined_nbrs = [basis] + list(filter(lambda x: x[0]==node and x!=basis, self.basis))      
        deg_n = len(coined_nbrs)
        coined_distr = [2 / deg_n  - 1] + [2 / (deg_n)] * (deg_n - 1)
        nbrs = []; distr = []
        for i in range(len(coined_nbrs)):
            shifted_nbrs = [coined_nbrs[i]] + list(filter(lambda x: x[1]==coined_nbrs[i][1] and x!=coined_nbrs[i], self.basis))      
            nbrs += shifted_nbrs
            deg_e = len(shifted_nbrs)
            distr += list(coined_distr[i] * np.array([2 / deg_e  - 1] + [2 / (deg_e)] * (deg_e - 1)))
        return distr, nbrs

    def signed_measure(self, basis_from:tuple[int,int], basis_to:tuple[int,int]):
        x, source_nbr = self.get_distr(basis_from)
        y, target_nbr = self.get_distr(basis_to)
        pops = [] 
        for i in range(len(source_nbr)):    
                if x[i] < 0:
                    target_nbr.append(source_nbr[i])
                    y.append(-x[i])
                    pops.append(i)
        for index in sorted(pops, reverse=True):
            del source_nbr[index]
            del x[index]
        pops = []
        for i in range(len(target_nbr)):    
                if y[i] < 0:
                    source_nbr.append(target_nbr[i])
                    x.append(-y[i])
                    pops.append(i)
        for index in sorted(pops, reverse=True):
            del target_nbr[index]
            del y[index]
        return (x, source_nbr), (y, target_nbr)       

    def ricci(self,basis_from:tuple[int,int], basis_to:tuple[int,int]):
        output={}
        (x, source_nbr), (y, target_nbr) = self.signed_measure(basis_from, basis_to)
        d=[[self.distance(snbr, tnbr) for tnbr in target_nbr] for snbr in source_nbr]
        output[basis_from,basis_to]= 1 - ot.emd2(np.array(x), np.array(y), d) / self.distance(basis_from,basis_to) 
        return output
    
    def arc_forward(self,source,target):
        return [(edge[0],edge[1]) for edge in self.graph.edges() if edge[0] == target and edge[1] !=source]

    def plot(self):
        drawing.draw(self.graph)

edges = [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
H = HyperGraph(edges)
# pprint(H.basis)
pprint(H.distance((2,1),(2,0)))
# pprint(H.get_distr((2,1)))
# pprint(H.ricci((2,1),(2,0)))
H.plot()
    
