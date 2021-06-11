import torch
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv


class GINEGNN(torch.nn.Module):
    def __init__(self, n_layers: int, vocab_size: int, emb_dim: int):
        super().__init__()

        layers = [torch.nn.Embedding(vocab_size, emb_dim, padding_idx=0)]
        torch.nn.init.xavier_uniform_(layers[0].weight.data)

        for i in range(n_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, 2 * emb_dim),
                torch.nn.BatchNorm1d(2 * emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * emb_dim, emb_dim),
            )
            layers.append(GINEConv(mlp))

        self._net = torch.nn.Sequential(*layers)

    def forward(self, data: Data):
        pass
