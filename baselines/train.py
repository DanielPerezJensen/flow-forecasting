# Machine learning imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from sklearn.metrics import mean_squared_error, r2_score

import wandb
import sklearn
import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

# Data processing imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Standard imports
import collections
import os
from tqdm import tqdm
import argparse

# Custom imports
import models
import data
import plotting
import utils

import numpy.typing as npt
from typing import Union, List, Optional, Any, Tuple

LoggerType = Union[Union[WandbLogger, TensorBoardLogger], bool]


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:

    summer_months_collection = []
    all_months_collection = []

    processed_path = os.path.join(get_original_cwd(), "data", "processed")

    dataset = data.RiverFlowDataset(
                root=processed_path,
                process=True,
                **cfg.data
            )

    # Split dataset into training, validation and test
    train, val, test = data.split_dataset(dataset, cfg.data,
                                          val_year_min=1999,
                                          val_year_max=2004,
                                          test_year_min=1974,
                                          test_year_max=1981)

    for seed in cfg.run.seeds:

        pl.seed_everything(seed, workers=True)

        # Convert Datasets to Dataloader to allow for training
        batch_size = cfg.training.batch_size

        train_loader = DataLoader(train, batch_size=batch_size,
                                  shuffle=True, num_workers=8)
        val_loader = DataLoader(val, batch_size=batch_size,
                                shuffle=False, num_workers=8)
        test_loader = DataLoader(test, batch_size=1,
                                 shuffle=False, num_workers=8)

        # Extract a sample and gather input and output dim from it
        sample = dataset[0]

        if cfg.model.name == "mlp":
            input_dim = torch.numel(sample[0])
        else:
            input_dim = sample[0].shape[1]

        output_dim = sample[1].shape[1]

        # Retrieve scaler used to scale values in dataset
        scaler = dataset.scaler

        if cfg.model.name == "mlp":
            model = models.MLP(input_dim, output_dim, scaler, cfg)
        elif cfg.model.name == "gru":
            model = models.GRU(input_dim, output_dim, scaler, cfg)
        elif cfg.model.name == "lstm":
            model = models.LSTM(input_dim, output_dim, scaler, cfg)

        # Set logger based on configuration file
        logger = set_logger(model, cfg)

        trainer = pl.Trainer(gpus=cfg.training.gpu, log_every_n_steps=1,
                             max_epochs=cfg.training.epochs, logger=logger)
        trainer.fit(model, train_loader, val_loader)

        if cfg.run.log and cfg.run.log.wandb:
            wandb.finish(quiet=True)

        # Save run in case config indicates it
        if cfg.run.save:
            summer_trgs, summer_preds = gather_preds(model, True, val, test)
            trgs, preds = gather_preds(model, False, val, test)

            summer_months_collection.append(
                np.stack((summer_trgs, summer_preds), axis=0)
            )
            all_months_collection.append(
                np.stack((trgs, preds), axis=0)
            )

    if cfg.run.save:
        save_dir = os.path.join(
            "..", "experiments", "baselines",
            (cfg.model.name + "-" + cfg.run.name)
        )
        os.makedirs(save_dir, exist_ok=True)

        summer_months = np.stack(summer_months_collection, axis=0)
        all_months = np.stack(all_months_collection, axis=0)

        with open(os.path.join(save_dir, "summer.npy"), "wb") as f:
            np.save(f, summer_months)

        with open(os.path.join(save_dir, "all.npy"), "wb") as f:
            np.save(f, all_months)


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


def gather_preds(
    model: nn.Module, summer: bool, *datasets: data.RiverFlowDataset
) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:

    new_dataset = data.RiverFlowDataset()

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

    return targets, predictions


if __name__ == "__main__":
    train()
