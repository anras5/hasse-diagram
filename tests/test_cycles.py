import numpy as np
import pytest

from src.hassediagram.cycles import _apply_transitive_reduction, _calculate_ranks, _find_and_merge_cycles


@pytest.mark.parametrize(
    "adjacency_matrix, labels, expected_matrix, expected_labels",
    [
        (
                np.array([
                    [0, 1, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]),
                {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
                np.array([
                    [0, 1, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]),
                {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
        ),
        (
                np.array([
                    [0, 1, 1, 1],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]),
                {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
                np.array([
                    [0, 1, 1],
                    [0, 0, 0],
                    [0, 0, 0]
                ]),
                {0: 'A, B', 1: 'C', 2: 'D'},
        ),
        (
                np.array([
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]
                ]),
                {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
                np.array([
                    [0, 0],
                    [0, 0],
                ]),
                {0: 'A, B', 1: 'C, D'},
        ),
        (
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 0, 0]
                ]),
                {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
                np.array([
                    [0]
                ]),
                {0: 'A, B, C, D'},
        ),
        (
                np.array([
                    [0]
                ]),
                {0: 'A'},
                np.array([
                    [0]
                ]),
                {0: 'A'},
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
                {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'},
                np.array([
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 0, 1, 1],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                ]),
                {0: 'A', 1: 'B', 2: 'C', 3: 'D, E', 4: 'F'},
        ),
    ],
    ids=[
        "no cycle",
        "one cycle",
        "two cycles",
        "one big cycle",
        "one node graph",
        "big graph"
    ]
)
def test_find_and_merge_cycles(adjacency_matrix, labels, expected_matrix, expected_labels):
    result_matrix, result_labels = _find_and_merge_cycles(adjacency_matrix, labels)

    assert np.array_equal(result_matrix, expected_matrix)
    assert result_labels == expected_labels


@pytest.mark.parametrize(
    "input_matrix, expected_matrix",
    [
        (
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ])
        ),
        (
                np.array([
                    [0, 1, 1, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ])
        ),
        (
                np.array([
                    [0, 1, 1, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ])
        ),
        (
                np.array([
                    [0, 1, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ])
        ),
    ],
    ids=[
        "no transitive edges",
        "single transitive edge",
        "double transitive edge",
        "multiple transitive edges"
    ]
)
def test_apply_transitive_reduction(input_matrix, expected_matrix):
    result_matrix = _apply_transitive_reduction(input_matrix.copy())

    assert np.array_equal(result_matrix, expected_matrix)


@pytest.mark.parametrize(
    "input_matrix, expected_ranks",
    [
        (
                np.array([
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                np.array([1, 2, 3, 4])
        ),
        (
                np.array([
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0]
                ]),
                np.array([1, 2, 3])
        ),
        (
                np.array([
                    [0, 0, 1, 0],
                    [1, 0, 0, 1],
                    [0, 0, 0, 1],
                    [0, 0, 0, 0]
                ]),
                np.array([2, 1, 3, 4])
        ),
        (
                np.array([
                    [0, 1, 1],
                    [0, 0, 1],
                    [0, 0, 0]
                ]),
                np.array([1, 2, 3])
        ),
        (
                np.array([
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]),
                np.array([1, 1, 1, 1])
        )
    ],
    ids=[
        "linear graph",
        "linear graph 2",
        "complex graph",
        "branching graph",
        "disconnected graph"
    ]
)
def test_calculate_ranks(input_matrix, expected_ranks):
    result_ranks = _calculate_ranks(input_matrix.copy())

    assert np.array_equal(result_ranks, expected_ranks)
