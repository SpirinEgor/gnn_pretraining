from typing import Dict, List, Iterator

import torch
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch import nn
from torch.nn import Parameter
from torch_geometric.data import Batch
from torchmetrics import MetricCollection, Metric

from src.models.gine_conv_pretraining import GINEConvPretraining
from src.models.modules.classifiers import TokensClassifier


class GINEConvTokenPrediction(GINEConvPretraining):
    def __init__(self, model_config: DictConfig, vocabulary_size: int, pad_idx: int, optim_config: DictConfig):
        super().__init__(model_config, vocabulary_size, pad_idx, optim_config)
        self.__tokens_classifier = TokensClassifier(model_config, vocabulary_size)

        self.__loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
        metrics: Dict[str, Metric] = {
            f"{holdout}_f1": SequentialF1Score(False, ignore_idx=[pad_idx]) for holdout in ["train", "val", "test"]
        }
        self.__metrics = MetricCollection(metrics)

        self.__pad_idx = pad_idx

    # ========== EXTENSION INTERFACE ==========

    def _get_parameters(self) -> List[Iterator[Parameter]]:
        return super()._get_parameters() + [self.__tokens_classifier.parameters()]

    def _shared_step(self, batched_graph: Batch, step: str) -> Dict:
        # [n nodes]
        tokens_mask = batched_graph["x_mask"]
        # [n masked nodes; max token parts]
        tokens_target = batched_graph["x_target"][tokens_mask]
        max_token_parts = tokens_target.shape[1]

        # [n masked nodes; hidden dim]
        encoded_graph = self._encoder(batched_graph)[tokens_mask]
        # [n masked nodes; vocabulary size]
        tokens_logits = self.__tokens_classifier(encoded_graph)

        # [n masked nodes; vocabulary size; max token parts]
        tokens_logits_ext = tokens_logits.unsqueeze(-1).expand(-1, -1, max_token_parts)
        loss = self.__loss(tokens_logits_ext, tokens_target)

        with torch.no_grad():
            # [n masked nodes; max tokens parts]
            _, tokens_pred = torch.topk(tokens_logits, max_token_parts, dim=-1)
            tokens_pred[tokens_target == self.__pad_idx] = self.__pad_idx

            metric: ClassificationMetrics = self.__metrics[f"{step}_f1"](tokens_pred.T, tokens_target.T)

        return {
            f"{step}/loss": loss,
            f"{step}/f1": metric.f1_score,
            f"{step}/precision": metric.precision,
            f"{step}/recall": metric.recall,
        }

    def _log_training_step(self, results: Dict):
        super()._log_training_step(results)
        self.log("f1", results["train/f1"], prog_bar=True, logger=False)

    def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT, step: str) -> Dict[str, torch.Tensor]:
        log = super()._prepare_epoch_end_log(step_outputs, step)
        metric: ClassificationMetrics = self.__metrics[f"{step}_f1"].compute()
        log.update(
            {f"{step}_f1": metric.f1_score, f"{step}_precision": metric.precision, f"{step}_recall": metric.recall}
        )
        return log
