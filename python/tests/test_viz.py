from pathlib import Path

import pytest
from graphviz import Digraph

import tensor.autograd.tensor_grad as tsg
from tensor import viz
import numpy as np


@pytest.mark.skipif(not viz.is_viz_available(), reason="Graphviz not installed")
def test_viz(tmp_path):
    x = tsg.var(np.random.randn(300, 2))
    w0 = tsg.var(np.random.randn(2, 100))
    b0 = tsg.var(np.random.randn(100))
    w1 = tsg.var(np.random.randn(100, 3))
    y = (x @ w0 + b0) @ w1

    graph_name = "test_viz"
    graph_path = tmp_path.joinpath(graph_name)
    graph: Digraph = viz.viz(y, filename=str(graph_path))

    graph.save()
    assert graph_path.exists()

    # graph.view()

