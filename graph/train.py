import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger

import wandb
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import os

import models
import data

from typing import List, Any, Union

import warnings

LoggerType = Union[Union[WandbLogger, TensorBoardLogger], bool]


# We ignore the GPU warning as it does not speed up graph nn at the moment
warnings.filterwarnings("ignore", ".*GPU available but not used.*")


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:

    pl.seed_everything(42, workers=True)

    processed_path = os.path.join(get_original_cwd(), "data", "processed")
    dataset = data.GraphFlowDataset(
                root=processed_path,
                process=True,
                **cfg.data
            )

    # Split dataset into training, validation and test
    train, val, test = data.split_dataset(dataset, freq=cfg.data.freq,
                                          val_year_min=1999,
                                          val_year_max=2004,
                                          test_year_min=1974,
                                          test_year_max=1981)

    train_loader = DataLoader(
        train, batch_size=cfg.training.batch_size, num_workers=8, shuffle=True
    )
    val_loader = DataLoader(
        val, batch_size=cfg.training.batch_size, num_workers=8
    )
    test_loader = DataLoader(
        test, batch_size=cfg.training.batch_size, num_workers=8
    )

    # Extract some information about the graph in our dataset
    data_sample = dataset[0]

    metadata = data_sample.metadata()

    # scaler = train.scaler
    scaler = dataset.scaler

    if cfg.data.sequential:
        model = models.HeteroSeqLSTM(cfg, metadata, scaler)

    # Dummy pass to initialize all layers
    with torch.no_grad():
        batch = next(iter(train_loader))
        if cfg.data.sequential:
            model(batch.xs_dict, batch.edge_indices_dict)
        else:
            model(batch.x_dict, batch.edge_index_dict)

    # Set logger based on configuration file
    logger = set_logger(model, cfg)

    trainer = pl.Trainer(gpus=cfg.training.gpu,
                         max_epochs=cfg.training.epochs,
                         deterministic=True,
                         logger=logger,
                         log_every_n_steps=1)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    if cfg.run.log and cfg.run.log.wandb:
        wandb.finish(quiet=True)


def set_logger(model: nn.Module, cfg: DictConfig) -> LoggerType:

    logger: LoggerType

    if cfg.run.log:
        if cfg.run.log.wandb:
            logger = WandbLogger(save_dir=get_original_cwd(),
                                 project=cfg.run.log.wandb.project,
                                 entity=cfg.run.log.wandb.entity,
                                 offline=cfg.run.log.wandb.offline)
            wandb_config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            logger.experiment.config.update(wandb_config)

            if cfg.run.log.wandb.watch:
                logger.watch(model, log="all")
        else:
            logger = TensorBoardLogger(save_dir=get_original_cwd())
    else:
        logger = False

    return logger


if __name__ == "__main__":
    train()
