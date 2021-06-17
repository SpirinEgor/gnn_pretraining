from typing import Dict, List, Iterator

import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.nn import Parameter
from torch_geometric.data import Batch
from torchmetrics import F1, MetricCollection, Metric

from src.data.graph import NodeType, EdgeType
from src.models.gine_conv_pretraining import GINEConvPretraining
from src.models.modules.classifiers import NodeTypeClassifier, EdgeTypeClassifier


class GINEConvTypeMasking(GINEConvPretraining):
    def __init__(self, model_config: DictConfig, vocabulary_size: int, pad_idx: int, optim_config: DictConfig):
        super().__init__(model_config, vocabulary_size, pad_idx, optim_config)

        self.__node_type_classifier = NodeTypeClassifier(model_config)
        self.__edge_type_classifier = EdgeTypeClassifier(model_config)

        self.__loss = nn.CrossEntropyLoss()
        metrics: Dict[str, Metric] = {}
        for holdout in ["train", "val", "test"]:
            metrics[f"{holdout}_node_type_f1"] = F1(len(NodeType))
            metrics[f"{holdout}_edge_type_f1"] = F1(len(EdgeType))
        self.__metrics = MetricCollection(metrics)

    # ========== EXTENSION INTERFACE ==========

    def _get_parameters(self) -> List[Iterator[Parameter]]:
        return super()._get_parameters() + [
            self.__node_type_classifier.parameters(),
            self.__edge_type_classifier.parameters(),
        ]

    def _shared_step(self, batched_graph: Batch, step: str) -> Dict:
        # [n nodes; hidden dim]
        encoded_graph = self._encoder(batched_graph)
        # [n nodes; number node types]
        node_type_logits = self.__node_type_classifier(encoded_graph)
        # [n nodes; number edge types]
        edge_type_logits = self.__edge_type_classifier(encoded_graph, batched_graph.edge_index)

        # [n nodes]
        node_mask = batched_graph["node_type_mask"]
        # float
        node_type_loss = self.__loss(node_type_logits[node_mask], batched_graph["node_type_target"][node_mask])

        # [n edges]
        edge_mask = batched_graph["edge_type_mask"]
        # float
        edge_type_loss = self.__loss(edge_type_logits[edge_mask], batched_graph["edge_type_target"][edge_mask])

        # TODO: we can use weighted sum here
        loss = node_type_loss + edge_type_loss

        with torch.no_grad():
            # [n nodes]
            node_type_pred = torch.argmax(node_type_logits, dim=-1)
            node_type_f1_score = self.__metrics[f"{step}_node_type_f1"](
                node_type_pred[node_mask], batched_graph["node_type_target"][node_mask]
            )
            # [n edges]
            edge_type_pred = torch.argmax(edge_type_logits, dim=-1)
            edge_type_f1_score = self.__metrics[f"{step}_edge_type_f1"](
                edge_type_pred[edge_mask], batched_graph["edge_type_target"][edge_mask]
            )

        return {
            f"{step}/loss": loss,
            f"{step}/f1-node type": node_type_f1_score,
            f"{step}/f1-edge type": edge_type_f1_score,
        }

    def _log_training_step(self, results: Dict):
        super()._log_training_step(results)
        self.log("f1-node type", results["train/f1-node type"], prog_bar=True, logger=False)
        self.log("f1-edge type", results["train/f1-edge type"], prog_bar=True, logger=False)

    def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT, step: str) -> Dict[str, torch.Tensor]:
        log = super()._prepare_epoch_end_log(step_outputs, step)
        log.update(
            {
                f"{step}_node_type_f1": self.__metrics[f"{step}_node_type_f1"].compute(),
                f"{step}_edge_type_f1": self.__metrics[f"{step}_edge_type_f1"].compute(),
            }
        )
        return log
