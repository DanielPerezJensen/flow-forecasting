import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

import wandb
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import os

import models
import data
import utils

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
        model = models.HeteroSeqGRU(cfg, metadata, scaler)
    else:
        model = models.HeteroMLP(cfg, metadata, scaler)

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

    if cfg.run.log and cfg.run.log.wandb:
        wandb.finish(quiet=True)

    # Save run in case config indicates it
    if cfg.run.save:
        final_val_rmse = trainer.logged_metrics["val_rmse"].item()
        save_dir = os.path.join(
            "..", "experiments", "graph", (
                cfg.model.name + "-" +
                str(int(cfg.data.ndsi.index)) + str(int(cfg.data.ndsi.surface)) +
                str(int(cfg.data.ndsi.cloud)) + str(int(cfg.data.ndvi.index)) +
                str(int(cfg.data.ndvi.surface)) + str(int(cfg.data.ndvi.cloud))
            )
        )
        os.makedirs(save_dir, exist_ok=True)

        summer_rmse = evaluate_preds(model, True, val, test)
        rmse = evaluate_preds(model, False, val, test)

        trainer.save_checkpoint(os.path.join(save_dir, "model.ckpt"))

        with open(os.path.join(save_dir, "summer-rmse.npy"), "wb") as f:
            np.save(f, summer_rmse)

        with open(os.path.join(save_dir, "rmse.npy"), "wb") as f:
            np.save(f, rmse)


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


def evaluate_preds(
    model: nn.Module, summer: bool, *datasets: data.GraphFlowDataset
) -> Any:

    new_dataset = data.GraphFlowDataset()

    for dataset in datasets:
        for date, value in dataset.data_date_dict.items():

            if summer:
                # Third month is split between winter and summer
                if int(date.astype(str).split("-")[1]) == 8:
                    new_dataset.set_data(date, value)
            else:
                new_dataset.set_data(date, value)

    loader = DataLoader(new_dataset, batch_size=1, num_workers=8)

    targets, predictions = utils.predict(model, loader)

    squared_error = ((targets - predictions) ** 2).mean(axis=(0, 1))
    rmse = np.sqrt(squared_error)

    return rmse


if __name__ == "__main__":
    train()
