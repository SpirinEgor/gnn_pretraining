from argparse import ArgumentParser
from os.path import join, basename

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import GraphDataModule
from src.data.vocabulary import Vocabulary
from src.models.gine_conv_masking_pretraining import GINEConvMaskingPretraining
from src.utils import filter_warnings


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    return arg_parser


def train(config_path: str):
    filter_warnings()
    config = OmegaConf.load(config_path)
    print(OmegaConf.to_yaml(config))
    seed_everything(config.seed, workers=True)

    # Load vocabulary
    vocabulary_path = join(config.data_folder, config.vocabulary.name)
    vocabulary = Vocabulary(vocabulary_path, config.vocabulary.n_tokens)

    # Init datamodule
    data_module = GraphDataModule(config.data_folder, vocabulary, config.data)

    # Init model
    model = GINEConvMaskingPretraining(config.model, len(vocabulary), vocabulary.pad[1], config.optimizer)

    # Define logger
    model_name = model.__class__.__name__
    dataset_name = basename(config.data_folder)
    wandb_logger = WandbLogger(project=f"{model_name} -- {dataset_name}", offline=config.offline)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename="{epoch:02d}-{val_loss:.4f}",
        period=config.train.save_every_epoch,
        save_top_k=-1,
    )
    early_stopping_callback = EarlyStopping(
        patience=config.train.patience, monitor="val_loss", verbose=True, mode="min"
    )
    lr_logger = LearningRateMonitor("step")

    gpu = 1 if torch.cuda.is_available() else None
    trainer = Trainer(
        max_epochs=config.train.n_epochs,
        gradient_clip_val=config.train.clip_norm,
        deterministic=True,
        check_val_every_n_epoch=config.train.val_every_epoch,
        log_every_n_steps=config.train.log_every_n_steps,
        logger=wandb_logger,
        gpus=gpu,
        progress_bar_refresh_rate=config.progress_bar_refresh_rate,
        callbacks=[lr_logger, early_stopping_callback, checkpoint_callback],
        resume_from_checkpoint=config.resume_from_checkpoint,
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test()


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    train(__args.config)
