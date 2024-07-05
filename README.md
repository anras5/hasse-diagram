# Hasse Diagram

This small package helps with plotting Hasse Diagrams and is very useful when presenting results for the MCDA methods.

## Installation

```
pip install hasse-diagram
```

## Example usage

### Networkx

```python
import numpy as np
from hassediagram import plot_hasse

data = np.array([
    [0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
labels = ["node a", "node b", "node c", "node d", "node e"]
plot_hasse(data, labels)
```

Result:

![img.png](./images/example_plot1.png)

- ### Graphviz dotstring

```python
import numpy as np
from hassediagram import hasse_graphviz

data = np.array([
    [0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])
labels = ["node a", "node b", "node c", "node d", "node e"]
print(hasse_graphviz(data, labels))
```

Result:

```
digraph {
	graph [bgcolor="#FFFFFF"]
	node [color="#E2E8F0" fontname="Segoe UI" fontsize="15 pt" style=filled]
	edge [arrowhead=vee color=black]
	compound=true
	node1 [label="node a"]
	node2 [label="node b, node c"]
	node3 [label="node d"]
	node4 [label="node e"]
	node1 -> node2
	node1 -> node3
	node2 -> node4
	subgraph cluster_1 {
		rank=same
		peripheries=0
	}
	subgraph cluster_2 {
		rank=same
		node1
		peripheries=0
	}
	subgraph cluster_3 {
		rank=same
		node2
		node3
		peripheries=0
	}
	subgraph cluster_4 {
		rank=same
		node4
		peripheries=0
	}
}
```

You can optionally turn off the transitive reduction and change the color of nodes and edges.

## Testing

```
pytest --cov=src --cov-report=term-missing
```

This package is inspired by a similar one for R: [hasseDiagram](https://github.com/kciomek/hasseDiagram)