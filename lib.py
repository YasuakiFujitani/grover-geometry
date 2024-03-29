from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
import ot
import numpy as np
from pprint import pprint
class ArcGraph():
    def __init__(self, edges: List[Tuple[int, int]]) -> None:
        self.graph = DiGraph(edges + [(b, a) for (a, b) in edges])

    def complete_graph(self, n: int) -> None:
        node = list(range(0, n))
        edges = [(a, b) for idx, a in enumerate(node) for b in node[idx + 1 :]]
        self.graph = DiGraph(edges + [(b, a) for (a, b) in edges])
        
    def distance(self, arc_from:Tuple[int,int], arc_to:Tuple[int,int]):
        #return 0 if arc_from == arc_to else nx.shortest_path_length(self.graph,arc_from[1], arc_to[0]) + 1
        if arc_from == arc_to:
            return 0
        else:
            if nx.shortest_path_length(self.graph,arc_from[1], arc_to[1]) <= nx.shortest_path_length(self.graph,arc_from[1], arc_to[0]):
                return nx.shortest_path_length(self.graph,arc_from[1], arc_to[1]) + 1
            else:
                return nx.shortest_path_length(self.graph,arc_from[1], arc_to[1])

    def get_distr(self, arc:Tuple[int,int]): 
        nbrs = list(filter(lambda x: x[0]==arc[1] and x[1]!=arc[0], self.graph.edges()))        
        return [2 / (len(nbrs) + 1)  - 1] + [2 / (len(nbrs) + 1)] * len(nbrs), [(arc[1],arc[0])] + nbrs

    def correlated_get_distr(self, arc:Tuple[int,int]): 
        nbrs = list(filter(lambda x: (x[0] == arc[1] or x[1] == arc[1]), self.graph.edges()))
        deg_v = len(list(filter(lambda x: x[0] == arc[1], self.graph.edges())))
        return [1/ ( 2 * deg_v)] * len(nbrs),  nbrs

    def signed_measure(self, source:Tuple[int,int], target:Tuple[int,int]):
        x, source_nbr = self.get_distr(source)
        y, target_nbr = self.get_distr(target)
        for i in range(len(source_nbr)-1,-1,-1):
                if x[i] < 0:
                    target_nbr.append(source_nbr.pop(i))
                    y.append(-x.pop(i))
        for i in range(len(target_nbr)-1,-1,-1):    
                if y[i] < 0:
                    source_nbr.append(target_nbr.pop(i))
                    x.append(-y.pop(i))
        return (x, source_nbr), (y, target_nbr)       

    def ricci(self, source:tuple[int,int], target:tuple[int,int]):
        (x, source_nbr), (y, target_nbr) = self.signed_measure(source, target)
        d=[[self.distance(snbr, tnbr) for tnbr in target_nbr] for snbr in source_nbr]
        output = 1 - ot.emd2(np.array(x), np.array(y), d) / self.distance(source,target) 
        return {(source,target):output}

    def correlated_ricci(self, source:Tuple[int,int], target:Tuple[int,int]):
        x, source_nbr = self.correlated_get_distr(source)
        y, target_nbr = self.correlated_get_distr(target)
        d=[[self.distance(snbr, tnbr) for tnbr in target_nbr] for snbr in source_nbr]
        output = 1 - ot.emd2(np.array(x), np.array(y), d) / self.distance(source,target) 
        return {(source,target):output}

    def plot(self):
        nx.draw(self.graph, with_labels=True, connectionstyle="arc3, rad = 0.1")
        plt.show()

#G=ArcGraph([(0,1),(1,2),(2,0)])
#print(G.correlated_get_distr((1,2)))
#G.complete_graph(4)
#print(G.ricci((1,2),(2,1)))
#print(G.correlated_ricci((1,2),(2,1)))
#print(G.ricci((0,1),(1,2)))
#x, source_nbr = G.correlated_get_distr((1,2))
#y, target_nbr = G.correlated_get_distr((2,1))
#d=[[G.distance(snbr, tnbr) for tnbr in target_nbr] for snbr in source_nbr]
#print(d)
