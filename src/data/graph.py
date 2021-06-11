from dataclasses import dataclass
from enum import Enum
from typing import List, Dict

import torch
from dpu_utils.codeutils import split_identifier_into_parts
from torch_geometric.data import Data

from src.data.vocabulary import Vocabulary


class NodeType(Enum):
    """Enum class to represent node type.
    - token nodes represent the raw lexemes in the program.
    - non-terminal nodes of the syntax tree.
    - vocabulary nodes that represents a subtoken,
        i.e. a word-like element which is retrieved by splitting an identifier into parts on camelCase or pascal_case.
    - symbol nodes that represent a unique symbol in the symbol table, such as a variable or function parameter.
    """

    TOKEN = 1
    NON_TERMINAL = 2
    VOCABULARY = 3
    SYMBOL = 4


@dataclass
class Node:
    """Class representing a node."""

    id: int
    token: str
    type: NodeType


class EdgeType(Enum):
    """Enum class to represent edge type.
    - NEXT: two consecutive token nodes.
    - CHILD: syntax nodes to their children nodes and tokens.
    - NEXT_USE: each token that is bound to a variable to all potential next uses of the variable.
    - LAST_LEXICAL_USE: each token that is bound to a variable to its last lexical use.
    - COMPUTED_FROM: the left hand side of an assignment expression to its right hand-side.
    - RETURNS_TO: all return/yield statements to the function declaration node where control returns.
    - OCCURRENCE_OF: all token and syntax nodes that bind to a symbol to the respective symbol node.
    - SUBTOKEN_OF: each identifier token node to the vocabulary nodes of its subtokens.
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

    id: int
    from_node: Node
    to_node: Node
    type: EdgeType


class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.__nodes = nodes
        self.__edges = edges

    @staticmethod
    def from_dict(graph_dict: Dict) -> "Graph":
        assert all(
            [key in graph_dict for key in ["nodes", "edges", "token-sequence", "supernodes"]]
        ), f"Incorrect graph structure: {graph_dict}"
        nodes = []
        edges = []
        for i, token in enumerate(graph_dict["nodes"]):
            if i in graph_dict["token-sequence"]:
                node_type = NodeType.TOKEN
            elif i in graph_dict["edges"]["SUBTOKEN_OF"]:
                node_type = NodeType.VOCABULARY
            elif i in graph_dict["supernodes"]:
                node_type = NodeType.SYMBOL
            else:
                node_type = NodeType.NON_TERMINAL
            nodes.append(Node(i, token, node_type))
        edge_id = 0
        for type_name, type_edges in graph_dict["edges"].items():
            edge_type = EdgeType[type_name]
            for root, children in type_edges.items():
                for child in children:
                    edges.append(Edge(edge_id, nodes[int(root)], nodes[int(child)], edge_type))
                    edge_id += 1
        return Graph(nodes, edges)

    @property
    def nodes(self) -> List[Node]:
        return self.__nodes

    @property
    def edges(self) -> List[Edge]:
        return self.__edges

    def to_torch(self, vocabulary: Vocabulary, max_token_parts: int) -> Data:
        """Convert this graph into torch-geometric graph

        :param vocabulary: vocabulary to convert token parts into ids
        :param max_token_parts: maximum number of token parts into tokenized version
        :return:
        """
        token = torch.full((len(self.__nodes), max_token_parts), vocabulary.pad[1], dtype=torch.long)
        for i, node in enumerate(self.__nodes):
            subtoken_ids = [vocabulary[st] for st in split_identifier_into_parts(node.token)[:max_token_parts]]
            token[i, : len(subtoken_ids)] = torch.tensor(subtoken_ids)

        node_type = torch.tensor([n.type.value for n in self.__nodes], dtype=torch.long)
        edge_index = torch.tensor(list(zip(*[[e.from_node.id, e.to_node.id] for e in self.__edges])), dtype=torch.long)
        edge_type = torch.tensor([e.type.value for e in self.__edges], dtype=torch.long)

        # save token to `x` so Data can calculate properties like `num_nodes`
        return Data(x=token, node_type=node_type, edge_index=edge_index, edge_type=edge_type)
