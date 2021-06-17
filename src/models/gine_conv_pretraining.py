from abc import abstractmethod
from typing import Dict, List, Iterator, Union

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.nn import Parameter
from torch_geometric.data import Batch, Data

from src.models.modules.gine_conv_encoder import GINEConvEncoder


class GINEConvPretraining(LightningModule):
    def __init__(self, model_config: DictConfig, vocabulary_size: int, pad_idx: int, optim_config: DictConfig):
        super().__init__()
        self.__optim_config = optim_config

        self._encoder = GINEConvEncoder(model_config, vocabulary_size, pad_idx)

    def forward(self, input_graph: Union[Data, Batch]):  # type: ignore
        return self._encoder(input_graph)

    def configure_optimizers(self) -> Dict:
        parameters = self._get_parameters()
        optimizer = torch.optim.AdamW(
            params=[{"params": p} for p in parameters],
            lr=self.__optim_config.lr,
            weight_decay=self.__optim_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: self.__optim_config.decay_gamma ** epoch
        )
        return {"optimizer": optimizer, "scheduler": scheduler}

    # ========== EXTENSION INTERFACE ==========

    def _get_parameters(self) -> List[Iterator[Parameter]]:
        return [self._encoder.parameters()]

    @abstractmethod
    def _shared_step(self, batched_graph: Batch, step: str) -> Dict:
        raise NotImplementedError()

    def _log_training_step(self, results: Dict):
        self.log_dict(results, on_step=True, on_epoch=False)

    def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT, step: str) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            losses = [so if isinstance(so, torch.Tensor) else so["loss"] for so in step_outputs]
            mean_loss = torch.stack(losses).mean()
        return {f"{step}_loss": mean_loss}

    # ========== MODEL STEP ==========

    def training_step(self, batched_graph: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        results = self._shared_step(batched_graph, "train")
        loss = results["train/loss"]
        self._log_training_step(results)
        return loss

    def validation_step(self, batched_graph: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        results = self._shared_step(batched_graph, "val")
        return results["val/loss"]

    def test_step(self, batched_graph: Batch, batch_idx: int) -> torch.Tensor:  # type: ignore
        results = self._shared_step(batched_graph, "test")
        return results["test/loss"]

    # ========== EPOCH END ==========

    def _shared_epoch_end(self, step_outputs: EPOCH_OUTPUT, step: str):
        log = self._prepare_epoch_end_log(step_outputs, step)
        self.log_dict(log, on_step=False, on_epoch=True)

    def training_epoch_end(self, training_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(training_step_output, "train")

    def validation_epoch_end(self, validation_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(validation_step_output, "val")

    def test_epoch_end(self, test_step_output: EPOCH_OUTPUT):
        self._shared_epoch_end(test_step_output, "test")
