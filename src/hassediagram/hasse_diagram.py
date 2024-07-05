from typing import List, Optional

import graphviz
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


def hasse_graphviz(
        data: np.ndarray,
        labels: Optional[List[str]] = None,
        transitive_reduction: bool = True,
        bg_color: str = '#FFFFFF',
        edge_color: str = 'black',
        node_color: str = '#E2E8F0'
) -> graphviz.Digraph:
    """
    Generates a Hasse diagram using Graphviz and returns it as a Digraph object.

    Parameters:
    - data (np.ndarray): Adjacency matrix of the graph.
    - labels (Optional[List[str]], optional): Labels for nodes.
    - transitive_reduction (bool, optional): Whether to perform transitive reduction.
    - bg_color (str, optional): Background color of the graph.
    - edge_color (str, optional): Color of the edges.
    - node_color (str, optional): Color of the nodes.

    Returns:
    - graphviz.Digraph: Graphviz Digraph object representing the Hasse diagram.
    """
    # Get matrix for the graph
    data, labels_dict, ranks = _hasse_matrix(data, labels, transitive_reduction)
    nr_nodes = data.shape[0]

    # Create digraph
    dot = graphviz.Digraph()

    # Setup general configuration for the graph
    dot.attr(compound='true')
    dot.graph_attr['bgcolor'] = bg_color
    dot.node_attr['style'] = 'filled'
    dot.node_attr['color'] = node_color
    dot.node_attr['fontname'] = 'Segoe UI'
    dot.node_attr['fontsize'] = '15 pt'
    dot.edge_attr['color'] = edge_color
    dot.edge_attr['arrowhead'] = 'vee'

    # Create nodes
    for i in range(nr_nodes):
        dot.node(f'node{i + 1}', label=str(labels_dict[i]))

    # Create edges
    for i in range(nr_nodes):
        for j in np.where(data[i, :])[0]:
            dot.edge(f'node{i + 1}', f'node{j + 1}')

    # Create sub graphs based on calculated ranks for each node
    max_rank = max(ranks)
    for rank in range(max_rank + 1):
        with dot.subgraph(name=f'cluster_{rank + 1}') as sub:
            sub.attr(rank='same')
            for i in range(nr_nodes):
                if ranks[i] == rank:
                    sub.node(f'node{i + 1}')
            sub.attr(peripheries='0')

    return dot


if __name__ == '__main__':
    data = np.array([
        [0, 1, 1, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    labels = ["node a", "node b", "node c", "node d", "node e"]
    print(hasse_graphviz(data, labels))
