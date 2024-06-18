from typing import Dict

import networkx as nx
import numpy as np


def _find_and_merge_cycles(adjacency_matrix: np.ndarray, labels: Dict[int, str]) -> tuple[np.ndarray, Dict[int, str]]:
    """
    Finds cycles in the graph represented by the adjacency matrix and merges nodes involved in each cycle.

    Parameters:
    - adjacency_matrix (np.ndarray): Adjacency matrix of the graph.
    - labels (Dict[int, str]): Mapping of node indices to their labels.

    Returns:
    - np.ndarray: Modified adjacency matrix after merging nodes in cycles.
    - Dict[int, str]: Updated labels after merging nodes in cycles.
    """
    G = nx.DiGraph(adjacency_matrix)
    # Find all cycles in the graph
    all_cycles = list(nx.simple_cycles(G))
    cycles = []
    for i in range(adjacency_matrix.shape[0]):
        cycles_with_node = sorted(filter(lambda x: i in x, all_cycles), key=lambda x: len(x), reverse=True)
        if len(cycles_with_node) > 0 and cycles_with_node[0] not in cycles:
            cycles.append(cycles_with_node[0])

    # Create a copy of the adjacency matrix to modify
    graph_copy = np.copy(adjacency_matrix)
    new_labels = labels

    # Merge nodes involved in each cycle into one node
    for cycle in cycles:
        # Create a combined label for the cycle nodes
        combined_label = ', '.join(map(str, [new_labels[i] for i in cycle]))

        # Merge all nodes in the cycle into the first node
        first_node = cycle[0]
        other_nodes = cycle[1:]

        # Merge nodes into first_node by removing them from the graph_copy
        for node in sorted(other_nodes):
            # Merge node into first_node (collapse the row and column)
            graph_copy[first_node, :] = np.logical_or(graph_copy[first_node, :], graph_copy[node, :])
            graph_copy[:, first_node] = np.logical_or(graph_copy[:, first_node], graph_copy[:, node])

        # Delete the row and column of the merged node
        graph_copy = np.delete(graph_copy, other_nodes, axis=0)
        graph_copy = np.delete(graph_copy, other_nodes, axis=1)

        # Update the labels
        new_labels[first_node] = combined_label
        new_labels = [new_labels[i] for i in range(len(new_labels)) if i not in other_nodes]
        new_labels = {i: label for i, label in enumerate(new_labels)}

    return graph_copy, new_labels


def _apply_transitive_reduction(data: np.ndarray) -> np.ndarray:
    """
    Applies transitive reduction to the given adjacency matrix.

    Parameters:
    - data (np.ndarray): Adjacency matrix.

    Returns:
    - np.ndarray: Transitive reduced adjacency matrix.
    """
    nr_nodes = data.shape[0]

    for source in range(nr_nodes):
        stack = list(np.where(data[source, :])[0])
        visited = np.zeros(nr_nodes, dtype=bool)
        visited[stack] = True

        while stack:
            element = stack.pop(0)
            children = np.where(data[element, :])[0]
            for child in children:
                data[source, child] = 0
                if not visited[child]:
                    stack.insert(0, child)
                    visited[child] = True

    return data
