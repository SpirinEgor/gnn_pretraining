from os.path import join

import torch
from omegaconf import OmegaConf
from torch_geometric.data import DataLoader

from src.data import Vocabulary, GraphDataset

RESOURCES_FOLDER = "test/resources"
VOCABULARY_FILE = join(RESOURCES_FOLDER, "vocabulary.pkl")
GRAPHS_FILE = join(RESOURCES_FOLDER, "graphs.jsonl.gz")
N_GRAPHS = 192
TEST_CONFIG = OmegaConf.load(join("configs", "dev.yaml"))


def test_multiple_workers():
    vocabulary = Vocabulary(VOCABULARY_FILE)
    train_dataset = GraphDataset(GRAPHS_FILE, vocabulary, TEST_CONFIG.data)
    data_loader = DataLoader(train_dataset, 128, num_workers=3)

    ids = []
    for graph in data_loader:
        ids.append(graph["id"])
    ids = torch.cat(ids).unique()
    assert ids.shape[0] == GRAPHS_FILE
