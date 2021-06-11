from argparse import ArgumentParser
from os.path import join

from omegaconf import OmegaConf

from src.data.datamodule import GraphDataModule
from src.data.vocabulary import Vocabulary


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    return arg_parser


def train(config_path: str):
    config = OmegaConf.load(config_path)

    # Load vocabulary
    vocabulary_path = join(config.data_folder, config.vocabulary.name)
    vocabulary = Vocabulary(vocabulary_path, config.vocabulary.n_tokens)

    # Init datamodule
    data_module = GraphDataModule(config.data_folder, vocabulary, config.data)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    train(__args.config)
