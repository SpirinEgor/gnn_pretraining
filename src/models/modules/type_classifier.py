import torch
from omegaconf import DictConfig
from torch import nn

from src.data.graph import NodeType, EdgeType


class NodeTypeClassifier(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.__relu = nn.ReLU()
        self.__node_type_linear = nn.Linear(config.hidden_dim, len(NodeType))

    def forward(self, encoded_graph: torch.Tensor) -> torch.Tensor:
        # Last layer of encoder is linear, we add ReLU to avoid merging linear operators
        # [n nodes; n types]
        logits = self.__node_type_linear(self.__relu(encoded_graph))
        return logits


class EdgeTypeClassifier(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.__relu = nn.ReLU()
        self.__edge_type_linear = nn.Linear(config.hidden_dim, len(EdgeType))

    def forward(self, encoded_graph: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Last layer of encoder is linear, we add ReLU to avoid merging linear operators
        # [n edges; hidden_dim]
        edge_features = encoded_graph[edge_index[0]] + encoded_graph[edge_index[1]]
        # [n edges; n types]
        logits = self.__edge_type_linear(self.__relu(edge_features))
        return logits
