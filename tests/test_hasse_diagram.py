import numpy as np
import pytest

from src.hassediagram.hasse_diagram import _hasse_matrix


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
