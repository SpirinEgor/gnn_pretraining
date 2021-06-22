from typing import Optional, List, Iterator, Dict

import torch
from commode_utils.loss import sequence_cross_entropy_loss
from commode_utils.metrics import SequentialF1Score, ClassificationMetrics
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from tokenizers import Tokenizer
from torch.nn import Parameter
from torch_geometric.data import Batch
from torchmetrics import MetricCollection

from src.models.gine_conv_pretraining import GINEConvPretraining
from src.utils import PAD, BOS, EOS


class GINEConvSequenceGenerating(GINEConvPretraining):
    def __init__(
        self,
        model_config: DictConfig,
        node_vocab_size: int,
        node_pad_idx: int,
        optim_config: DictConfig,
        label_tokenizer: Tokenizer,
        teacher_forcing: float = 0.0,
        pretrain: Optional[str] = None,
    ):
        super().__init__(model_config, node_vocab_size, node_pad_idx, optim_config)
        if pretrain is not None:
            state_dict = torch.load(pretrain)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            self._encoder.load_state_dict(state_dict)

        self.__pad_idx = label_tokenizer.token_to_id(PAD)
        decoder_step = LSTMDecoderStep(model_config.decoder, label_tokenizer.get_vocab_size(), self.__pad_idx)
        self._decoder = Decoder(decoder_step, label_tokenizer.get_vocab_size(), self.token_to_id(BOS), teacher_forcing)

        ignore_idx = [label_tokenizer.token_to_id(it) for it in [PAD, BOS, EOS]]
        self.__metric_dict = MetricCollection(
            {
                holdout: SequentialF1Score(mask_after_pad=True, pad_idx=self.__pad_idx, ignore_idx=ignore_idx)
                for holdout in ["train", "val", "test"]
            }
        )

    # ========== EXTENSION INTERFACE ==========

    def _get_parameters(self) -> List[Iterator[Parameter]]:
        return super()._get_parameters() + [self._decoder.parameters()]

    def _shared_step(self, batched_graph: Batch, step: str) -> Dict:
        # [n nodes; hidden dim]
        encoded_graph = self._encoder(batched_graph)
        graph_sizes = [it.num_nodes for it in batched_graph.to_data_list()]

        # [max seq len; batch size]
        target = batched_graph["target"]

        # [max seq len; batch size; vocab size]
        logits = self._decoder(encoded_graph, graph_sizes, target.shape[0], target)

        loss = sequence_cross_entropy_loss(logits, target, self.__pad_idx)

        with torch.no_grad():
            # [max seq len; batch size]
            prediction = logits.argmax(-1)

            metric: ClassificationMetrics = self.__metrics[f"{step}_f1"](prediction, target)

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
