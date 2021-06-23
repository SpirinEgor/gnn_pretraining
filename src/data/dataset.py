import gzip
import json
from os.path import dirname, basename
from typing import Iterator, Optional, Union, Dict

import torch
from omegaconf import DictConfig
from tokenizers import Tokenizer
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data

from src.data.graph import Graph, NodeType, EdgeType
from src.utils import PAD, MASK


class GraphDataset(IterableDataset):

    __known_dataset_stats = {
        "dev": {"train": 552, "val": 185, "test": 192},
        "small": {"train": 44_683, "val": 14_892, "test": 14_934},
        "full": {"train": 56_666_194, "val": 19_892_270, "test": 18_464_490},
        "xglue-code-to-text-python": {"train": 249_686, "val": 13_774, "test": 14_761},
        "xglue-code-to-text-python-dev": {"train": 1_000, "val": 1_000, "test": 1_000},
    }

    def __init__(self, graph_filepath: str, tokenizer: Tokenizer, config: DictConfig):
        super().__init__()
        self.__graph_filepath = graph_filepath
        self.__config = config
        self.__tokenizer = tokenizer
        self.__pad_idx = tokenizer.token_to_id(PAD)
        self.__tokenizer.enable_padding(pad_id=self.__pad_idx, pad_token=PAD, length=config.max_token_parts)
        self.__tokenizer.enable_truncation(max_length=config.max_token_parts)

    def __iter__(self) -> Iterator[Data]:
        worker_info = get_worker_info()
        n_workers = 1 if worker_info is None else worker_info.num_workers
        take_each_point = 0 if worker_info is None else worker_info.id
        with gzip.open(self.__graph_filepath, "rb") as input_file:
            for i, line in enumerate(input_file):
                if i % n_workers != take_each_point:
                    continue
                raw_graph = json.loads(line.decode("utf-8"))
                graph = Graph.from_dict(raw_graph)
                if not self._validate(graph):
                    continue
                graph = graph.to_torch(self.__tokenizer)
                graph["id"] = i
                self._extract_label(graph, raw_graph)
                yield graph

    def _extract_label(self, graph: Data, raw_graph: Dict):
        if self.__config.task.name == "type masking":
            self._type_masking_task(graph)
        elif self.__config.task.name == "token prediction":
            self._token_prediction_task(graph)
        elif self.__config.task.name == "sequence generating":
            self._sequence_generating_task(graph, raw_graph)
        else:
            raise ValueError(f"Unknown task for graph learning: {self.__config.task.name}")

    def __len__(self) -> Optional[int]:
        dataset_name = basename(dirname(self.__graph_filepath))
        if dataset_name in self.__known_dataset_stats:
            holdout_name = basename(self.__graph_filepath).split(".", 1)[0].rsplit("_", 1)[1]
            return self.__known_dataset_stats[dataset_name][holdout_name]
        return None

    @staticmethod
    def _mask_property(graph: Data, attr_name: str, p: float, mask_value: Union[int, torch.Tensor]):
        target, mask = f"{attr_name}_target", f"{attr_name}_mask"
        graph[target] = graph[attr_name].clone().detach()
        graph[mask] = torch.rand(graph[target].shape[0]) < p
        graph[attr_name][graph[mask]] = mask_value

    def _type_masking_task(self, graph: Data):
        self._mask_property(graph, "node_type", self.__config.task.p_node, len(NodeType))
        self._mask_property(graph, "edge_type", self.__config.task.p_edge, len(EdgeType))

    def _token_prediction_task(self, graph: Data):
        mask_value = torch.tensor(self.__tokenizer.encode(MASK).ids, dtype=torch.long)
        self._mask_property(graph, "x", self.__config.task.p, mask_value)

    def _sequence_generating_task(self, graph: Data, raw_graph: Dict):
        graph["target"] = raw_graph[self.__config.task.field]

    def _validate(self, graph: Graph) -> bool:
        if self.__config.max_n_nodes is None:
            return True
        return len(graph.nodes) <= self.__config.max_n_nodes
