from dataclasses import dataclass
from enum import Enum
from typing import List, Dict


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
    group: NodeType


class EdgeType(Enum):
    """Enum class to represent edge type.
    - NEXT: two consecutive token nodes.
    - CHILD: syntax nodes to their children nodes and tokens.
    - NEXT_USE: each token that is bound to a variable to all potential next uses of the variable.
    - NEXT_LEXICAL_USE: each token that is bound to a variable to its next lexical use.
    - LAST_LEXICAL_USE: each token that is bound to a variable to its last lexical use.
    - COMPUTED_FROM: the left hand side of an assignment expression to its right hand-side.
    - RETURNS_TO: all return/ yield statements to the function declaration node where control returns.
    - OCCURRENCE_OF: all token and syntax nodes that bind to a symbol to the respective symbol node.
    - SUBTOKEN_OF: each identifier token node to the vocabulary nodes of its subtokens.
    """

    NEXT = 0
    CHILD = 1
    NEXT_USE = 2
    NEXT_LEXICAL_USE = 3
    LAST_LEXICAL_USE = 4
    COMPUTED_FROM = 5
    RETURNS_TO = 6
    OCCURRENCE_OF = 7
    SUBTOKEN_OF = 8


@dataclass
class Edge:
    """Class representing an edge."""

    id: int
    from_node: Node
    to_node: Node
    edge_type: EdgeType


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
                    edges.append(Edge(edge_id, nodes[root], nodes[child], edge_type))
                    edge_id += 1
        return Graph(nodes, edges)

    @property
    def nodes(self) -> List[Node]:
        return self.__nodes

    @property
    def edges(self) -> List[Edge]:
        return self.__edges
