from typing import Optional, List, Iterator, Dict

import torch
from commode_utils.losses import SequenceCrossEntropyLoss
from commode_utils.modules import LSTMDecoderStep, Decoder
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.processors import TemplateProcessing
from torch.nn import Parameter
from torch_geometric.data import Batch
from torchmetrics import MetricCollection

from src.metric import CodeXGlueBleu
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
            print("Use pretrained weights for sequence generating model")
            state_dict = torch.load(pretrain)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            state_dict = {k.removeprefix("_encoder."): v for k, v in state_dict.items() if k.startswith("_encoder.")}
            self._encoder.load_state_dict(state_dict)
        else:
            print("No pre-trained weights for sequence generating model")

        self.__pad_idx = label_tokenizer.token_to_id(PAD)
        self.__label_tokenizer = label_tokenizer
        self.__label_tokenizer.add_special_tokens([BOS, EOS])
        self.__label_tokenizer.post_processor = TemplateProcessing(
            single=f"{BOS} $A {EOS}",
            pair=None,
            special_tokens=[
                (BOS, self.__label_tokenizer.token_to_id(BOS)),
                (EOS, self.__label_tokenizer.token_to_id(EOS)),
            ],
        )
        self.__label_tokenizer.enable_padding(pad_id=self.__pad_idx, pad_token=PAD)
        self.__label_tokenizer.decoder = ByteLevel()

        decoder_step = LSTMDecoderStep(model_config.decoder, label_tokenizer.get_vocab_size(), self.__pad_idx)
        self._decoder = Decoder(
            decoder_step, label_tokenizer.get_vocab_size(), label_tokenizer.token_to_id(BOS), teacher_forcing
        )

        self.__metric_dict = MetricCollection(
            {f"{holdout}_bleu": CodeXGlueBleu(4, 1) for holdout in ["train", "val", "test"]}
        )

        self.__loss = SequenceCrossEntropyLoss(self.__pad_idx)
        self.__temperature = model_config.temperature

    # ========== EXTENSION INTERFACE ==========

    def _get_parameters(self) -> List[Iterator[Parameter]]:
        return super()._get_parameters() + [self._decoder.parameters()]

    def _shared_step(self, batched_graph: Batch, step: str) -> Dict:
        # [n nodes; hidden dim]
        encoded_graph = self._encoder(batched_graph)
        graph_sizes = torch.tensor(
            [it.num_nodes for it in batched_graph.to_data_list()], dtype=torch.int, device=self.device
        )

        # [batch size]
        target = batched_graph["target"]
        target_tokenized = self.__label_tokenizer.encode_batch(target)
        # [max seq len; batch size]
        target_ids = torch.empty(
            (len(target_tokenized[0]), len(target_tokenized)), dtype=torch.long, device=self.device
        )
        for i, tt in enumerate(target_tokenized):
            target_ids[:, i] = torch.tensor(tt.ids)

        # [max seq len; batch size; vocab size]
        if step == "train":
            logits = self._decoder(encoded_graph, graph_sizes, target_ids.shape[0], target_ids)
        else:
            logits = self._decoder(encoded_graph, graph_sizes, target_ids.shape[0])
        logits /= self.__temperature

        loss = self.__loss(logits, target_ids)

        with torch.no_grad():
            # [batch size, max seq len]
            prediction = logits.detach().argmax(-1).T.tolist()
            # [batch size]
            predicted_sequence = self.__label_tokenizer.decode_batch(prediction)
            if step != "train":
                print(predicted_sequence[0])

            bleu = self.__metric_dict[f"{step}_bleu"](predicted_sequence, target)

        return {f"{step}/loss": loss, f"{step}/bleu": bleu}

    def _log_training_step(self, results: Dict):
        super()._log_training_step(results)
        self.log("bleu", results["train/bleu"], prog_bar=True, logger=False)

    def _prepare_epoch_end_log(self, step_outputs: EPOCH_OUTPUT, step: str) -> Dict[str, torch.Tensor]:
        log = super()._prepare_epoch_end_log(step_outputs, step)
        log[f"{step}_bleu"] = self.__metric_dict[f"{step}_bleu"].compute()
        return log
