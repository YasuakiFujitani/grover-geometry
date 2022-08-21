import networkx as nx
import numpy as np
import ot

def get_distr(node):  
    nbrs = list(G.neighbors(node))
    return [0] + [1 / len(nbrs)] * len(nbrs), [node] + nbrs       
    
def compute_ricci_curvature(G: nx.Graph):
    output={}
    for edge in G.edges():
        x, source_nbr = get_distr(edge[0])
        y, target_nbr = get_distr(edge[1])
        apsp=dict(nx.all_pairs_dijkstra_path_length(G))
        d=[[apsp[snbr][tnbr] for tnbr in target_nbr] for snbr in source_nbr ]
        output[(edge[0],edge[1])]= 1 - ot.emd2(np.array(x), np.array(y), d) 
    nx.set_edge_attributes(G, output, "ricci")
    return output

G=nx.DiGraph([[1,0],[0,1],[0,3],[1,2],[2,4],[4,1],[1,5],[3,2],[5,4],[6,5],[1,6],[2,6]])
compute_ricci_curvature(G)
for n1,n2 in list(G.edges):
        print("Ricci curvature of edge (%s,%s) is %f" % (n1 ,n2, G[n1][n2]["ricci"]))


