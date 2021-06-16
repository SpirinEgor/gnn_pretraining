from typing import Dict, Union

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch_geometric.data import Data, Batch
from torchmetrics import F1, MetricCollection

from src.data.graph import NodeType, EdgeType
from src.models.modules.gine_conv_encoder import GINEConvEncoder
from src.models.modules.type_classifier import NodeTypeClassifier, EdgeTypeClassifier


class GINEConvMaskingPretraining(LightningModule):
    def __init__(self, model_config: DictConfig, vocabulary_size: int, pad_idx: int, optim_config: DictConfig):
        super().__init__()
        self.__optim_config = optim_config

        self.__encoder = GINEConvEncoder(model_config, vocabulary_size, pad_idx)
        self.__node_type_classifier = NodeTypeClassifier(model_config)
        self.__edge_type_classifier = EdgeTypeClassifier(model_config)

        self.__loss = nn.CrossEntropyLoss()
        metrics = {}
        for holdout in ["train", "val", "test"]:
            metrics[f"{holdout}_node_type_f1"] = F1(len(NodeType))
            metrics[f"{holdout}_edge_type_f1"] = F1(len(EdgeType))
        self.__metrics = MetricCollection(metrics)

    def forward(self, input_graph: Union[Data, Batch]):  # type: ignore
        return self.__encoder(input_graph)

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(
            params=[
                {"params": self.__encoder.parameters()},
                {"params": self.__node_type_classifier.parameters()},
                {"params": self.__edge_type_classifier.parameters()},
            ],
            lr=self.__optim_config.lr,
            weight_decay=self.__optim_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: self.__optim_config.decay_gamma ** epoch
        )
        return {"optimizer": optimizer, "scheduler": scheduler}

    # ========== MODEL STEP ==========

    def _shared_step(self, batched_graph: Batch, step: str) -> Dict:
        # [n nodes; hidden dim]
        encoded_graph = self.__encoder(batched_graph)
        node_type_logits = self.__node_type_classifier(encoded_graph)
        edge_type_logits = self.__edge_type_classifier(encoded_graph, batched_graph.edge_index)

        # [n nodes]
        node_mask = batched_graph["node_mask"]
        # int
        node_type_loss = self.__loss(node_type_logits[node_mask], batched_graph["node_target"][node_mask])

        # [n edges]
        edge_mask = batched_graph["edge_mask"]
        # int
        edge_type_loss = self.__loss(edge_type_logits[edge_mask], batched_graph["edge_target"][edge_mask])

        # TODO: we can use weighted sum here
        loss = node_type_loss + edge_type_loss

        with torch.no_grad():
            # [n nodes]
            node_type_pred = torch.argmax(node_type_logits, dim=-1)
            node_type_f1_score = self.__metrics[f"{step}_node_type_f1"](node_type_pred, batched_graph["node_target"])
            # [n edges]
            edge_type_pred = torch.argmax(edge_type_logits, dim=-1)
            edge_type_f1_score = self.__metrics[f"{step}_edge_type_f1"](edge_type_pred, batched_graph["edge_target"])

        return {
            f"{step}/loss": loss,
            f"{step}/f1-node type": node_type_f1_score,
            f"{step}/f1-edge type": edge_type_f1_score,
        }

    def training_step(self, batched_graph: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        result = self._shared_step(batched_graph, "train")
        loss = result["train/loss"]
        self.log_dict(result, on_step=True, on_epoch=False)
        self.log("f1-node type", result["train/f1-node type"], prog_bar=True, logger=False)
        self.log("f1-edge type", result["train/f1-edge type"], prog_bar=True, logger=False)
        return loss

    def validation_step(self, batched_graph: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        result = self._shared_step(batched_graph, "val")
        return result["val/loss"]

    def test_step(self, batched_graph: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        result = self._shared_step(batched_graph, "test")
        return result["test/loss"]

    # ========== EPOCH END ==========

    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, step: str):
        with torch.no_grad():
            losses = [so if isinstance(so, torch.Tensor) else so["loss"] for so in step_outputs]
            mean_loss = torch.stack(losses).mean()
        total_node_type_f1 = self.__metrics[f"{step}_node_type_f1"].compute()
        total_edge_type_f1 = self.__metrics[f"{step}_edge_type_f1"].compute()
        log = {
            f"{step}_loss": mean_loss,
            f"{step}_node_type_f1": total_node_type_f1,
            f"{step}_edge_type_f1": total_edge_type_f1,
        }
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, training_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(training_step_output, "train")

    def validation_epoch_end(self, validation_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(validation_step_output, "val")

    def test_epoch_end(self, test_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(test_step_output, "test")
