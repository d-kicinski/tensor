from collections import defaultdict
from typing import Dict
import subprocess

import tensor.autograd.tensor_grad as tsg

from graphviz import Digraph


def viz(variable: tsg.Variable, filename: str) -> Digraph:
    f = Digraph(filename=filename)
    f.attr(rankdir='LR')
    f.attr('node', shape='rectangle')

    def loop(var: tsg.Variable):
        if var.op:
            f.node(str(id(var)), f"{str(var)}\n{var.value.shape}")
            f.node(str(id(var.op)), str(var.op), shape="oval")
            f.edge(str(id(var.op)), str(id(var)))

            if inputs := var.op.inputs:
                i: tsg.Variable
                for i in inputs:
                    f.node(str(id(i)), f"{str(i)}\n{i.value.shape}")
                    f.edge(str(id(i)), str(id(var.op)))
                    loop(i)
    loop(variable)

    return f


def is_viz_available() -> bool:
    ret = subprocess.call(["which", "dot"])
    if ret != 0:
        return False
    return True

