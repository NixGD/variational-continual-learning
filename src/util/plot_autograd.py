"""
    File name: plot-pytorch-autograd-graph.py
    Author: Ludovic Trottier
    Date created: November 8, 2017.
    Date last modified: November 8, 2017
    Credits: moskomule (https://discuss.pytorch.org/t/print-autograd-graph/692/15)
"""

# DISCLAIMER
# This code was not written by members of the AML project group, and neither is there
# any intention to make it appear as if though it had been. The code is open source
# and free to use. For detailed credits, view the link in the documentation comment.


from graphviz import Digraph
import torch
from torch.autograd import Variable


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph.

    Blue nodes are trainable Variables (weights, bias).
    Orange node are saved tensors for the backward pass.

    Args:
        var: output Variable
        params: list of (name, Parameters)
    """

    param_map = {id(v): k for k, v in params}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')

    dot = Digraph(
        filename='network',
        format='pdf',
        node_attr=node_attr,
        graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(variable):
        if variable not in seen:

            node_id = str(id(variable))

            if torch.is_tensor(variable):
                node_label = "saved tensor\n{}".format(tuple(variable.size()))
                dot.node(node_id, node_label, fillcolor='orange')

            elif hasattr(variable, 'variable'):
                variable_name = param_map.get(id(variable.variable))
                variable_size = tuple(variable.variable.size())
                node_name = "{}\n{}".format(variable_name, variable_size)
                dot.node(node_id, node_name, fillcolor='lightblue')

            else:
                node_label = type(variable).__name__.replace('Backward', '')
                dot.node(node_id, node_label)

            seen.add(variable)

            if hasattr(variable, 'next_functions'):
                for u in variable.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(variable)))
                        add_nodes(u[0])

            if hasattr(variable, 'saved_tensors'):
                for t in variable.saved_tensors:
                    dot.edge(str(id(t)), str(id(variable)))
                    add_nodes(t)

    add_nodes(var.grad_fn)

    return dot


if __name__ == '__main__':
    from torchvision import models

    inputs = torch.randn(1, 3, 224, 224)
    resnet18 = models.resnet18()
    y = resnet18(Variable(inputs))

    g = make_dot(y, resnet18.named_parameters())
    g.view()