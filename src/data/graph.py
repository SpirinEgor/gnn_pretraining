from dataclasses import dataclass
from enum import Enum
from itertools import chain
from typing import List, Dict

import torch
from pyvis.network import Network
from tokenizers import Tokenizer
from torch_geometric.data import Data


class NodeType(Enum):
    """Enum class to represent node type.
    — token nodes represent the raw lexemes in the program.
    — non-terminal nodes of the syntax tree.
    — vocabulary nodes that represents a subtoken,
        i.e., a word-like element that retrieved by splitting an identifier into parts on camelCase or pascal_case.
    — symbol nodes that represent a unique symbol in the symbol table, such as a variable or function parameter.
    """

    TOKEN = 0
    NON_TERMINAL = 1
    VOCABULARY = 2
    SYMBOL = 3


@dataclass
class Node:
    """Class representing a node."""

    id: int
    token: str
    type: NodeType


class EdgeType(Enum):
    """Enum class to represent edge type.
    — NEXT: two consecutive token nodes.
    — CHILD: syntax nodes to their children nodes and tokens.
    — NEXT_USE: each token that bound to a variable to all potential next uses of the variable.
    — LAST_LEXICAL_USE: each token that bound to a variable to its last lexical use.
    — COMPUTED_FROM: the left-hand side of an assignment expression to its right hand-side.
    — RETURNS_TO: all return/yield statements to the function declaration node where control returns.
    — OCCURRENCE_OF: all token and syntax nodes that bind to a symbol to the respective symbol node.
    — SUBTOKEN_OF: each identifier token node to the vocabulary nodes of its subtokens.
    """

    NEXT = 0
    CHILD = 1
    NEXT_USE = 2
    LAST_LEXICAL_USE = 3
    COMPUTED_FROM = 4
    RETURNS_TO = 5
    OCCURRENCE_OF = 6
    SUBTOKEN_OF = 7


@dataclass
class Edge:
    """Class representing an edge."""

    from_node: Node
    to_node: Node
    type: EdgeType


class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.__nodes = nodes
        self.__edges = edges

    @staticmethod
    def get_node_type(graph_dict: Dict, node_id: int) -> NodeType:
        node_type = NodeType.NON_TERMINAL
        if node_id in graph_dict["edges"]["SUBTOKEN_OF"]:
            node_type = NodeType.VOCABULARY
        elif node_id in graph_dict["edges"]["OCCURRENCE_OF"]:
            node_type = NodeType.SYMBOL
        elif node_id in graph_dict["token-sequence"]:
            node_type = NodeType.TOKEN
        return node_type

    @staticmethod
    def from_dict(graph_dict: Dict) -> "Graph":
        assert all(
            [key in graph_dict for key in ["nodes", "edges", "token-sequence", "supernodes"]]
        ), f"Incorrect graph structure: {graph_dict}"
        edges: List[Edge] = []
        for key in ["SUBTOKEN_OF", "OCCURRENCE_OF"]:
            graph_dict["edges"][key] = {int(k): v for k, v in graph_dict["edges"].get(key, {}).items()}
        nodes = [Node(i, token, Graph.get_node_type(graph_dict, i)) for i, token in enumerate(graph_dict["nodes"])]
        for type_name, type_edges in graph_dict["edges"].items():
            edge_type = EdgeType[type_name]
            edges.extend(
                chain(
                    *(
                        [Edge(nodes[int(root)], nodes[int(child)], edge_type) for child in children]
                        for root, children in type_edges.items()
                    )
                )
            )
        return Graph(nodes, edges)

    @property
    def nodes(self) -> List[Node]:
        return self.__nodes

    @property
    def edges(self) -> List[Edge]:
        return self.__edges

    def to_torch(self, tokenizer: Tokenizer) -> Data:
        """Convert this graph into torch-geometric graph

        :param tokenizer: tokenizer to convert token parts into ids
        :return:
        """
        node_tokens = [n.token for n in self.nodes]
        encoded = tokenizer.encode_batch(node_tokens)
        token = torch.tensor([enc.ids for enc in encoded], dtype=torch.long)

        node_type = torch.tensor([n.type.value for n in self.__nodes], dtype=torch.long)
        edge_index = torch.tensor(list(zip(*[[e.from_node.id, e.to_node.id] for e in self.__edges])), dtype=torch.long)
        edge_type = torch.tensor([e.type.value for e in self.__edges], dtype=torch.long)

        # save token to `x` so Data can calculate properties like `num_nodes`
        return Data(x=token, node_type=node_type, edge_index=edge_index, edge_type=edge_type)

    def draw(self, height: int = 1000, width: int = 1000, notebook: bool = True) -> Network:
        """Visualize graph using [pyvis](https://pyvis.readthedocs.io/en/latest/) library

        :param graph: graph instance to visualize
        :param height: height of target visualization
        :param width: width of target visualization
        :param notebook: pass True if visualization should be displayed in notebook
        :return: pyvis Network instance
        """
        net = Network(height=height, width=width, directed=True, notebook=notebook)
        net.barnes_hut(gravity=-10000, overlap=1, spring_length=1)

        for node in self.nodes:
            net.add_node(node.id, label=node.token, group=node.type.value, title=f"id: {node.id}\ntoken: {node.token}")

        for edge in self.edges:
            net.add_edge(edge.from_node.id, edge.to_node.id, label=edge.type.name, group=edge.type.value)

        return net
