from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph
import ot
import numpy as np
from pprint import pprint
class ArcGraph():
    graph = DiGraph()

    def __init__(self, edges: List[Tuple[int, int]]) -> None:
        self.graph = DiGraph(edges + [(b, a) for (a, b) in edges])

    def complete_graph(self, n: int) -> None:
        node = list(map(str, range(0, n)))
        edges = [(a, b) for idx, a in enumerate(node) for b in node[idx + 1 :]]
        self.graph = DiGraph(edges + [(b, a) for (a, b) in edges])
        
    def distance(self, arc_from:Tuple[int,int], arc_to:Tuple[int,int]):
        return 0 if arc_from == arc_to else nx.shortest_path_length(self.graph,arc_from[1], arc_to[0]) + 1

    def get_distr(self, source, target): 
        nbrs = list(filter(lambda x: x[0]==target and x[1]!=source, self.graph.edges()))        
        return [2 / (len(nbrs) + 1)  - 1] + [2 / (len(nbrs) + 1)] * len(nbrs), [(target,source)] + nbrs

    def signed_measure(self, source, target, target_forward):
        x, source_nbr = self.get_distr(source, target)
        y, target_nbr = self.get_distr(target, target_forward)
        for i in range(len(source_nbr)):    
                if x[i] < 0:
                    target_nbr.append(source_nbr.pop(i))
                    y.append(-x.pop(i))
                    break
        for i in range(len(target_nbr)):    
                if y[i] < 0:
                    source_nbr.append(target_nbr.pop(i))
                    x.append(-y.pop(i))
                    break
        return (x, source_nbr), (y, target_nbr)       

    def ricci_reverse(self):
        output={}
        for edge in self.graph.edges():
            (x, source_nbr), (y, target_nbr) = self.signed_measure(edge[0], edge[1], edge[0])
            d=[[self.distance(snbr, tnbr) for tnbr in target_nbr] for snbr in source_nbr]
            output[(edge[0],edge[1]),(edge[1],edge[0])]= 1 - ot.emd2(np.array(x), np.array(y), d) / self.distance((edge[0],edge[1]),(edge[1],edge[0])) 
        return output

    def ricci_forward(self):
        output={}
        for edge in self.graph.edges():
            arc_forward = self.arc_forward(edge[0],edge[1])
            for (_,target_forward) in arc_forward:
                (x, source_nbr),(y, target_nbr) = self.signed_measure(edge[0],edge[1],target_forward)
                d=[[self.distance(snbr,tnbr) for tnbr in target_nbr] for snbr in source_nbr]
                output[(edge[0],edge[1]),(edge[1],target_forward)]= 1 - ot.emd2(np.array(x), np.array(y), d) / self.distance((edge[0],edge[1]),(edge[1],edge[0])) 
        return output
    
    def arc_forward(self,source,target):
        return [(edge[0],edge[1]) for edge in self.graph.edges() if edge[0] == target and edge[1] !=source]

    def plot(self):
        nx.draw(self.graph, with_labels=True, connectionstyle="arc3, rad = 0.1")
        plt.show()

G=ArcGraph([(1,2),(2,3),(3,4),(4,5)])
G.complete_graph(3)
pprint(G.ricci_reverse())
# print(G.ricci_forward())
