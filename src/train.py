from argparse import ArgumentParser
from os.path import join, basename

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins import DDPPlugin
from tokenizers import Tokenizer

from src.data.datamodule import GraphDataModule
from src.models.gine_conv_token_prediction import GINEConvTokenPrediction
from src.models.gine_conv_type_masking import GINEConvTypeMasking
from src.utils import filter_warnings, PAD


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    return arg_parser


def train(config_path: str):
    filter_warnings()
    config = OmegaConf.load(config_path)
    print(OmegaConf.to_yaml(config))
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

    # Define logger
    model_name = model.__class__.__name__
    dataset_name = basename(config.data_folder)
    wandb_logger = WandbLogger(project=f"{model_name} -- {dataset_name}", offline=config.offline)

    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        filename="{epoch:02d}-{step:02d}-{val_loss:.4f}",
        monitor="val_loss",
        every_n_val_epochs=1,
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
        val_check_interval=config.train.val_every_step,
        log_every_n_steps=config.train.log_every_n_steps,
        logger=wandb_logger,
        gpus=gpu,
        progress_bar_refresh_rate=config.progress_bar_refresh_rate,
        callbacks=[lr_logger, early_stopping_callback, checkpoint_callback],
        resume_from_checkpoint=config.resume_from_checkpoint,
        plugins=DDPPlugin(find_unused_parameters=False),
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    train(__args.config)
