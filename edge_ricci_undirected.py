# This program is aimed to calculate the Ricci curvature on edges defined in the following paper.
# Taiki Yamada "AN ESTIMATE OF THE FIRST NON-ZERO EIGENVALUE OF THE LAPLACIAN BY THE RICCI CURVATURE ON EDGES OF GRAPHS", Osaka Journal of Mathematics. 57(1) (2020) P.151-P.163

from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import ot
import numpy as np
import itertools


class UndirectedEdgeGraph:
    def __init__(self, edges: List[Tuple[int, int]]) -> None:
        self.graph = nx.Graph(edges)

    def check_connection(self, edge_1, edge_2):
        c = 0
        if edge_1[0] == edge_2[0] or (
            edge_1[0] == edge_2[1] or (edge_1[1] == edge_2[0] or edge_1[1] == edge_2[1])
        ):
            c = 1
        return c

    def distance_between_edges(self, edge_1, edge_2):
        G = self.graph
        a1 = nx.shortest_path_length(G, edge_1[0], edge_2[0])
        a2 = nx.shortest_path_length(G, edge_1[0], edge_2[1])
        a3 = nx.shortest_path_length(G, edge_1[1], edge_2[0])
        a4 = nx.shortest_path_length(G, edge_1[1], edge_2[1])
        inverse_edge_2 = (edge_2[1], edge_2[0])
        if edge_1 == edge_2:
            return 0
        elif edge_1 == inverse_edge_2:
            return 0
        else:
            return min(a1, a2, a3, a4) + 1

    def get_distribution(self, the_edge):
        G = self.graph
        nbrs = []
        for edge in G.edges():
            if self.check_connection(edge, the_edge):
                nbrs.append(edge)
        inverse_the_edge = (the_edge[1], the_edge[0])
        # treating
        for edge in nbrs:
            if edge == the_edge:
                nbrs.remove(edge)
            elif edge == inverse_the_edge:
                nbrs.remove(edge)
        # this is a treatment for the case such that the_edge is not included in G.edges() but inverse_the_edge is included
        return [0] + [1 / len(nbrs)] * len(nbrs), [the_edge] + nbrs

    # G = nx.complete_graph(3)
    # get_distribution((1,2),G)
    # ([0, 0.5, 0.5], [(2, 1), (0, 1), (0, 2)])

    def select_connected_edge_pair(self):
        G = self.graph
        pairs = itertools.combinations(G.edges, 2)
        connected_pairs = []
        for pair in pairs:
            if self.check_connection(pair[0], pair[1]):
                connected_pairs.append(pair)
        return list(connected_pairs)

    def compute_ricci_curvature_between_edge_undirected(self):
        output = {}
        pairs = self.select_connected_edge_pair()
        for pair in pairs:
            source = pair[0]
            target = pair[1]
            source_nbr_distribution, source_nbr = self.get_distribution(source)
            target_nbr_distribution, target_nbr = self.get_distribution(target)
            cost_matrix_between_neighbors = [
                [self.distance_between_edges(snbr, tnbr) for snbr in source_nbr]
                for tnbr in target_nbr
            ]
            output[(source, target)] = 1 - ot.emd2(
                np.array(source_nbr_distribution),
                np.array(target_nbr_distribution),
                cost_matrix_between_neighbors,
            ) / self.distance_between_edges(source, target)
        return output

    def plot(self):
        nx.draw(self.graph, with_labels=True, connectionstyle="arc3, rad = 0.1")
        plt.show()
