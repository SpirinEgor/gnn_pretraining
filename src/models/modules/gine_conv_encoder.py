import torch
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv
from torch_sparse import SparseTensor

from src.data.graph import NodeType, EdgeType


def gine_conv_nn(in_dim: int, out_dim: int) -> nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, 2 * in_dim),
        torch.nn.BatchNorm1d(2 * in_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(2 * in_dim, out_dim),
    )


class GINEConvEncoder(torch.nn.Module):
    """Predict node and edge types using GINEConv operator."""

    def __init__(self, config: DictConfig, vocabulary_size: int, pad_idx: int):
        super().__init__()

        self.__pad_idx = pad_idx
        self.__st_embedding = nn.Embedding(vocabulary_size, config.embed_dim, padding_idx=pad_idx)
        # Additional embedding value for masked token
        self.__node_type_embedding = nn.Embedding(len(NodeType) + 1, config.embed_dim)
        self.__edge_type_embedding = nn.Embedding(len(EdgeType) + 1, config.embed_dim)

        for emb_layer in [self.__st_embedding, self.__node_type_embedding, self.__edge_type_embedding]:
            torch.nn.init.xavier_uniform_(emb_layer.weight.data)

        self.__gine_conv_start = GINEConv(gine_conv_nn(config.embed_dim, config.hidden_dim))
        self.__hidden_gine_conv = nn.ModuleList(
            [GINEConv(gine_conv_nn(config.hidden_dim, config.hidden_dim)) for _ in range(config.n_hidden_layers)]
        )

    def forward(self, batched_graph: Batch) -> torch.Tensor:
        # [n nodes]
        n_parts = (batched_graph.x != self.__pad_idx).sum(dim=-1).reshape(-1, 1)
        # There are some nodes without token, e.g., `s = ""` would lead to node for "" with empty token.
        not_empty_mask = (n_parts != 0).reshape(-1)
        # [n nodes; embed dim]
        subtokens_embed = self.__st_embedding(batched_graph.x).sum(dim=1)
        subtokens_embed[not_empty_mask] /= n_parts[not_empty_mask]

        # [n nodes; embed dim]
        node_types_embed = self.__node_type_embedding(batched_graph["node_type"])
        # [n nodes; embed dim]
        node_embedding = subtokens_embed + node_types_embed

        # [n edges; embed dim]
        edge_embedding = self.__edge_type_embedding(batched_graph["edge_type"])

        # Sparse adjacent matrix
        num_nodes = batched_graph.num_nodes
        adj_t = SparseTensor.from_edge_index(batched_graph.edge_index, edge_embedding, (num_nodes, num_nodes)).t()
        adj_t = adj_t.device_as(edge_embedding)

        # [n nodes; hidden dim]
        gine_features = self.__gine_conv_start(x=node_embedding, edge_index=adj_t)
        for hidden_gine_conv_layer in self.__hidden_gine_conv:
            gine_features = hidden_gine_conv_layer(x=gine_features, edge_index=adj_t)
        return gine_features
