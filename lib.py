from typing import List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from networkx import DiGraph


class ArcGraph:
    graph = DiGraph()

    def __init__(self, edges: List[Tuple[int, int]]) -> None:
        self.graph = DiGraph(edges + [(b, a) for (a, b) in edges])

    def complete_graph(self, n: int) -> DiGraph:
        node = list(map(str, range(1, n + 1)))
        self.graph = DiGraph()
        self.graph.add_nodes_from(node)
        self.graph.add_edges_from(
            [(a, b) for idx, a in enumerate(node) for b in node[idx + 1 :]]
        )
        self.graph.add_edges_from(
            [(b, a) for idx, a in enumerate(node) for b in node[idx + 1 :]]
        )

    def distance(
        self, arc_from: List[Tuple[int, int]], arc_to: Tuple[int, int]
    ) -> DiGraph:
        return nx.shortest_path_length(self.graph, arc_from[1], arc_to[0]) + 1

    def plot(self):
        nx.draw(self.graph, with_labels=True, connectionstyle="arc3, rad = 0.1")
        plt.show()
