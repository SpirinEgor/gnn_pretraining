import json
from typing import Iterator

from omegaconf import DictConfig
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data

from src.data.graph import Graph
from src.data.vocabulary import Vocabulary


class GraphDataset(IterableDataset):
    def __init__(self, graph_filepath: str, vocabulary: Vocabulary, config: DictConfig):
        super().__init__()
        self.__graph_filepath = graph_filepath
        self.__config = config
        self.__vocabulary = vocabulary

    def __iter__(self) -> Iterator[Data]:
        worker_info = get_worker_info()
        if worker_info is not None:
            raise RuntimeError("Graph dataset does not support multiple workers")
        with open(self.__graph_filepath, "r") as input_file:
            for line in input_file:
                raw_graph = json.loads(line)
                graph = Graph.from_dict(raw_graph).to_torch(self.__vocabulary, self.__config.max_token_parts)
                yield graph
