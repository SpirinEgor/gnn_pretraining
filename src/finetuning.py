from argparse import ArgumentParser
from os.path import join
from typing import cast

from commode_utils.common import print_config
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from tokenizers import Tokenizer

from src.data.datamodule import GraphDataModule
from src.models.gine_conv_sequence_generating import GINEConvSequenceGenerating
from src.train import train
from src.utils import filter_warnings, PAD


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    return arg_parser


def fine_tune(config_path: str):
    filter_warnings()
    config = cast(DictConfig, OmegaConf.load(config_path))
    print_config(config, ["data", "model", "optimizer", "train"])
    seed_everything(config.seed, workers=True)

    # Load node tokenizer
    node_tokenizer_path = join(config.data_folder, config.node_tokenizer)
    node_tokenizer = Tokenizer.from_file(node_tokenizer_path)

    # Load label tokenizer
    label_tokenizer_path = join(config.data_folder, config.label_tokenizer)
    label_tokenizer = Tokenizer.from_file(label_tokenizer_path)

    # Init datamodule
    data_module = GraphDataModule(config.data_folder, node_tokenizer, config.data)

    # Init model
    model = GINEConvSequenceGenerating(
        config.model,
        node_tokenizer.get_vocab_size(),
        node_tokenizer.token_to_id(PAD),
        config.optimizer,
        label_tokenizer,
        config.train.teacher_forcing,
    )

    train(model, data_module, config)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    fine_tune(__args.config)
