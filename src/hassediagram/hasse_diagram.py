from typing import List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .cycles import _apply_transitive_reduction, _calculate_ranks, _find_and_merge_cycles


def _hasse_matrix(data: np.ndarray, labels: Optional[List[str]] = None, transitive_reduction: bool = True):
    """
    Constructs the Hasse diagram matrix and optionally applies transitive reduction.

    Parameters:
    - data (np.ndarray): Adjacency matrix of the graph.
    - labels (Optional[List[str]], optional): Labels for nodes.
    - transitive_reduction (bool, optional): Whether to perform transitive reduction.

    Returns:
    - np.ndarray: Processed adjacency matrix.
    - List[str]: Processed labels.
    - np.ndarray: Ranks for the nodes in the graph.
    """
    assert isinstance(data, np.ndarray)
    assert data.shape[0] > 0
    assert data.shape[0] == data.shape[1]

    nr_nodes = data.shape[0]

    if labels is None:
        labels = [f"a{i}" for i in range(1, nr_nodes + 1)]
    else:
        assert len(labels) == nr_nodes

    labels_dict = {i: labels[i] for i in range(nr_nodes)}

    np.fill_diagonal(data, 0)
    data, labels_dict = _find_and_merge_cycles(data, labels_dict)

    if transitive_reduction:
        data = _apply_transitive_reduction(data)

    ranks = _calculate_ranks(data)

    return data, labels_dict, ranks


def plot_hasse(
        data: np.ndarray,
        labels: Optional[List[str]] = None,
        transitive_reduction: bool = True,
        edge_color: str = 'black',
        node_color: str = 'none'
):
    """
    Generates and displays a Hasse diagram based on given parameters.

    Parameters:
    - data (np.ndarray): Adjacency matrix of the graph.
    - labels (Optional[List[str]], optional): Labels for nodes.
    - transitive_reduction (bool, optional): Whether to perform transitive reduction.
    - edge_color (str, optional): Color of edges.
    - node_color (str, optional): Color of nodes.
    """

    data, labels_dict, ranks = _hasse_matrix(data, labels, transitive_reduction)
    nr_nodes = data.shape[0]

    G = nx.DiGraph()
    for i in range(nr_nodes):
        G.add_node(i, label=labels_dict[i], rank=ranks[i])

    for i in range(nr_nodes):
        for j in np.where(data[i, :])[0]:
            G.add_edge(i, j)

    pos = nx.multipartite_layout(G, subset_key="rank")
    pos = {k: (v[1], -v[0]) for k, v in pos.items()}

    node_size = 2000
    for node, (x, y) in pos.items():
        pos[node] = (x - 0.5 * node_size, y - 0.5 * node_size)

    # Draw nodes and edges
    nx.draw(
        G,
        pos,
        labels=nx.get_node_attributes(G, "label"),
        node_shape="s",
        edge_color=edge_color,
        node_color=node_color,
        font_color='white',
        arrows=True,
        arrowsize=15,
        node_size=1200,
        bbox=dict(facecolor="teal", edgecolor='black', boxstyle='round,pad=0.2')
    )

    plt.axis('off')
    plt.show()
