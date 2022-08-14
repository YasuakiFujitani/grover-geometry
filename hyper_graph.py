import matplotlib.pyplot as plt
import hypernetx as hnx
from networkx import fruchterman_reingold_layout as layout


def hyper():
    scenes = [
        ("A", "B"),
        ("B", "C"),
        ("C", "D", "F"),
        ("C", "E", "D", "A"),
        ("D", "F", "G", "B", "A", "K", "H"),
        ("B", "F"),
        ("A", "D"),
        ("B", "G"),
    ]
    H = hnx.Hypergraph(scenes)
    hnx.drawing.draw(H)
