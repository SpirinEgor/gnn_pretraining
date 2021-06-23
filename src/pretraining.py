from argparse import ArgumentParser
from os.path import join
from typing import cast

from commode_utils.common import print_config
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import seed_everything
from tokenizers import Tokenizer

from src.data.datamodule import GraphDataModule
from src.models.gine_conv_token_prediction import GINEConvTokenPrediction
from src.models.gine_conv_type_masking import GINEConvTypeMasking
from src.train import train
from src.utils import filter_warnings, PAD


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    return arg_parser


def pretrain(config_path: str):
    filter_warnings()
    config = cast(DictConfig, OmegaConf.load(config_path))
    print_config(config, ["data", "model", "optimizer", "train"])
    seed_everything(config.seed, workers=True)

    # Load tokenizer
    tokenizer_path = join(config.data_folder, config.tokenizer)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id(PAD)

    # Init datamodule
    data_module = GraphDataModule(config.data_folder, tokenizer, config.data)

    # Init model
    task_name = config.data.task.name
    if task_name == "type masking":
        model = GINEConvTypeMasking(config.model, vocab_size, pad_idx, config.optimizer)
    elif task_name == "token prediction":
        model = GINEConvTokenPrediction(config.model, vocab_size, pad_idx, config.optimizer)
    else:
        print(f"Unknown pretraining task: {task_name}")
        return

    train(model, data_module, config)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    pretrain(__args.config)
