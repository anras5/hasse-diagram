import graphviz
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.hassediagram.hasse_diagram import _hasse_matrix, hasse_graphviz, plot_hasse

matplotlib.use('Agg')


@pytest.mark.parametrize(
    "input_matrix, labels, transitive_reduction, expected_matrix, expected_labels, expected_ranks",
    [
        (
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                None,
                True,
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                {0: 'a1', 1: 'a2', 2: 'a3', 3: 'a4'},
                np.array([1, 2, 3, 4])
        ),
        (
                np.array([
                    [0, 1, 1, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                ["A", "B", "C", "D"],
                True,
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
                np.array([1, 2, 3, 4])
        ),
        (
                np.array([
                    [0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0]
                ]),
                ["X", "Y", "Z"],
                False,
                np.array([
                    [0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0]
                ]),
                {0: 'X', 1: 'Y', 2: 'Z'},
                np.array([1, 2, 3])
        ),
        (
                np.array([
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 1, 1],
                    [1, 1, 0, 0, 1, 0],
                    [1, 1, 0, 1, 0, 0],
                    [1, 1, 0, 0, 0, 0],
                ]),
                ['A', 'B', 'C', 'D', 'E', 'F'],
                True,
                np.array([
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ]),
                {0: 'A', 1: 'B', 2: 'C', 3: 'D, E', 4: 'F'},
                np.array([3, 4, 1, 2, 2])
        )
    ],
    ids=[
        "simple_no_transitive_edges",
        "with_transitive_edges",
        "without_transitive_reduction",
        "big_graph"
    ]
)
def test_hasse_matrix(input_matrix, labels, transitive_reduction, expected_matrix, expected_labels, expected_ranks):
    result_matrix, result_labels, result_ranks = _hasse_matrix(input_matrix.copy(), labels, transitive_reduction)

    assert np.array_equal(result_matrix, expected_matrix)
    assert result_labels == expected_labels
    assert np.array_equal(result_ranks, expected_ranks)


@pytest.mark.filterwarnings("ignore:.*FigureCanvasAgg is non-interactive.*")
@pytest.mark.parametrize(
    "input_matrix, labels, transitive_reduction, edge_color, node_color",
    [
        (
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                None,
                True,
                'black',
                'none'
        ),
        (
                np.array([
                    [0, 1, 1, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                ["A", "B", "C", "D"],
                True,
                'blue',
                'red'
        ),
        (
                np.array([
                    [0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0]
                ]),
                ["X", "Y", "Z"],
                False,
                'green',
                'yellow'
        ),
        (
                np.array([
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]),
                None,
                True,
                'cyan',
                'magenta'
        )
    ],
    ids=[
        "simple_no_labels",
        "with_labels_transitive_reduction",
        "without_transitive_reduction",
        "disconnected_graph"
    ]
)
def test_plot_hasse(input_matrix, labels, transitive_reduction, edge_color, node_color):
    plt.figure()
    plot_hasse(
        data=input_matrix,
        labels=labels,
        transitive_reduction=transitive_reduction,
        edge_color=edge_color,
        node_color=node_color
    )
    plt.close()


@pytest.mark.parametrize(
    "input_matrix, labels, transitive_reduction, bg_color, edge_color, node_color, expected_nodes, expected_edges",
    [
        (
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                None,
                True,
                '#FFFFFF',
                'black',
                '#E2E8F0',
                ['node1', 'node2', 'node3', 'node4'],
                [('node1', 'node2'), ('node2', 'node3'), ('node3', 'node4')]
        ),
        (
                np.array([
                    [0, 1, 1, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                ["A", "B", "C", "D"],
                True,
                '#000000',
                'blue',
                'red',
                ['node1', 'node2', 'node3', 'node4'],
                [('node1', 'node2'), ('node3', 'node4'), ('node2', 'node3')]
        ),
        (
                np.array([
                    [0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0]
                ]),
                ["X", "Y", "Z"],
                False,
                '#FFFFFF',
                'green',
                'yellow',
                ['node1', 'node2', 'node3'],
                [('node1', 'node2'), ('node1', 'node3'), ('node2', 'node3')]
        ),
        (
                np.array([
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]),
                None,
                True,
                '#123456',
                'cyan',
                'magenta',
                ['node1', 'node2', 'node3', 'node4'],
                []
        )
    ],
    ids=[
        "simple_no_labels",
        "with_labels_transitive_reduction",
        "without_transitive_reduction",
        "disconnected_graph"
    ]
)
def test_hasse_graphviz(input_matrix, labels, transitive_reduction, bg_color, edge_color, node_color, expected_nodes,
                        expected_edges):
    dot = hasse_graphviz(
        data=input_matrix,
        labels=labels,
        transitive_reduction=transitive_reduction,
        bg_color=bg_color,
        edge_color=edge_color,
        node_color=node_color
    )

    # Check if the output is a graphviz.Digraph object
    assert isinstance(dot, graphviz.Digraph)

    # Check if all nodes are present in the dot string
    lines = dot.body
    for node_label in labels or expected_nodes:
        assert any(node_label in node for node in lines)

    # Check edges
    lines = dot.body
    for edge_start, edge_end in expected_edges:
        assert any(f'{edge_start} -> {edge_end}' in line for line in lines)
