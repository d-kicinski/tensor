from collections import defaultdict
from typing import Dict
import subprocess

import tensor.autograd.tensor_grad as tsg

from graphviz import Digraph


def viz(variable: tsg.Variable, filename: str) -> Digraph:
    f = Digraph(filename=filename)
    f.attr(rankdir='LR')

    f.attr('node', shape='rectangle')

    register: Dict[str, int] = defaultdict(int)

    def loop(var: tsg.Variable):
        if var.op:

            var_id = f"{str(var)} {var.value.shape} #{register[str(var)]}"
            op_id = f"{str(var.op)} #{register[str(var.op)]}"

            register[str(var)] += 1
            register[str(var.op)] += 1

            f.node(var_id, f"{str(var)}\n{var.value.shape}")
            f.node(op_id, str(var.op), shape="oval")
            f.edge(op_id, var_id)

            var.op.backward(var.grad)
            if inputs := var.op.inputs:
                i: tsg.Variable
                for i in inputs:

                    var_id = f"{str(i)} {i.value.shape} #{register[str(i)]}"

                    f.node(var_id, f"{str(i)}\n{i.value.shape}")
                    f.edge(var_id, op_id)
                    loop(i)
    loop(variable)

    return f


def is_viz_available() -> bool:
    ret = subprocess.call(["which", "dot"])
    if ret != 0:
        return False
    return True

