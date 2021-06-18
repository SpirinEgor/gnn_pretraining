from os.path import join

import torch
from omegaconf import OmegaConf
from tokenizers import Tokenizer
from torch_geometric.data import DataLoader

from src.data import GraphDataset

RESOURCES_FOLDER = "test/resources"
TOKENIZER_FILE = join(RESOURCES_FOLDER, "tokenizer.json")
GRAPHS_FILE = join(RESOURCES_FOLDER, "graphs.jsonl.gz")
N_GRAPHS = 192
TEST_CONFIG = OmegaConf.load(join("test", "resources", "config.yaml"))


def test_multiple_workers():
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    train_dataset = GraphDataset(GRAPHS_FILE, tokenizer, TEST_CONFIG.data)
    data_loader = DataLoader(train_dataset, 128, num_workers=3)

    ids = []
    for graph in data_loader:
        ids.append(graph["id"])
    ids = torch.cat(ids).unique()
    assert ids.shape[0] == N_GRAPHS
