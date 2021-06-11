import gzip
import json
from typing import Iterator

import torch
from omegaconf import DictConfig
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data

from src.data.graph import Graph
from src.data.vocabulary import Vocabulary


class GraphDataset(IterableDataset):

    __MASK_VALUE = -1

    def __init__(self, graph_filepath: str, vocabulary: Vocabulary, config: DictConfig):
        super().__init__()
        self.__graph_filepath = graph_filepath
        self.__config = config
        self.__vocabulary = vocabulary

    def __iter__(self) -> Iterator[Data]:
        worker_info = get_worker_info()
        if worker_info is not None:
            raise RuntimeError("Graph dataset does not support multiple workers")
        with gzip.open(self.__graph_filepath, "rb") as input_file:
            for line in input_file:
                raw_graph = json.loads(line.decode("utf-8"))
                graph = Graph.from_dict(raw_graph).to_torch(self.__vocabulary, self.__config.max_token_parts)
                if self.__config.task.name == "masking":
                    self._mask_graph_task(graph)
                else:
                    raise ValueError(f"Unknown task for graph learning: {self.__config.task.name}")
                yield graph

    def _mask_type(self, graph: Data, attr_name: str, p: float):
        target, mask, attr_type = f"{attr_name}_target", f"{attr_name}_mask", f"{attr_name}_type"
        graph[target] = graph[attr_type].clone().detach()
        graph[mask] = torch.rand(graph[target].shape[0]) < p
        graph[attr_type][graph[mask]] = self.__MASK_VALUE

    def _mask_graph_task(self, graph: Data):
        self._mask_type(graph, "node", self.__config.task.p_node)
        self._mask_type(graph, "edge", self.__config.task.p_edge)
