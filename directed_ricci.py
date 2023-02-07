import itertools
import networkx as nx
import numpy as np
import ot
import itertools

alpha = 0.9999999
def get_distribution(node, G: nx.DiGraph):
    nbrs = list(G.neighbors(node))
    return [alpha] + [(1.0 - alpha) / len(nbrs)] * len(nbrs), [node] + nbrs


def distance_between_nodes(node_1, node_2, G: nx.DiGraph):
    return nx.shortest_path_length(G, node_1, node_2)


def compute_ricci_curvature_between_nodes_directed(G: nx.DiGraph):
    output = {}
    # We consider the case when every edge is connected by one of directed path
    pairs = list(itertools.combinations(G.nodes(), 2))
    for pair in pairs:
        source = pair[0]
        target = pair[1]
        source_nbr_distribution, source_nbr = get_distribution(source, G)
        target_nbr_distribution, target_nbr = get_distribution(target, G)
        cost_matrix_between_neighbors = [
            [distance_between_nodes(snbr, tnbr, G) for snbr in source_nbr]
            for tnbr in target_nbr
        ]
        output[(source, target)] = (
            1
            - ot.emd2(
                np.array(source_nbr_distribution),
                np.array(target_nbr_distribution),
                cost_matrix_between_neighbors,
            )
            / distance_between_nodes(source, target, G)
        ) / (1 - alpha)
    return output


#G = [(0,1),(1,2),(2,3),(3,4),(4,0)]
#G_D = nx.DiGraph(G)
#print(compute_ricci_curvature_between_nodes_directed(G_D))