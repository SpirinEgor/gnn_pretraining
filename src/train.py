from os.path import basename

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger


def train(model: LightningModule, data_module: LightningDataModule, config: DictConfig):
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
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model)
