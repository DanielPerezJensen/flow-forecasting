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

    summer_months_collection = []
    all_months_collection = []

    rmses = []
    nses = []

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

    for seed in cfg.run.seeds:
        pl.seed_everything(seed, workers=True)

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

        in_channels_dict = {}

        if cfg.data.sequential:
            for key, value in data_sample.to_dict().items():
                if isinstance(key, str) and isinstance(value, dict):
                    in_channels_dict[key] = value["xs"].shape[-1]

            model = models.HeteroSeqGRU(in_channels_dict, cfg, metadata, scaler)
        else:
            for key, value in data_sample.to_dict().items():
                if isinstance(key, str) and isinstance(value, dict):
                    in_channels_dict[key] = value["x"].shape[-1]

            model = models.HeteroMLP(in_channels_dict, cfg, metadata, scaler)

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
                             logger=logger,
                             log_every_n_steps=1)

        trainer.fit(model, train_loader, val_loader)

        # Save run in case config indicates it
        if cfg.run.save:

            rmses.append(trainer.logged_metrics["val_rmse"].item())
            nses.append(trainer.logged_metrics["val_nse"].item())

            summer_trgs, summer_preds = gather_preds(model, True, val, test)
            trgs, preds = gather_preds(model, False, val, test)

            summer_months_collection.append(
                np.stack((summer_trgs, summer_preds), axis=0)
            )
            all_months_collection.append(
                np.stack((trgs, preds), axis=0)
            )

        if cfg.run.log and cfg.run.log.wandb:
            wandb.finish(quiet=True)

    if cfg.run.save:
        save_dir = os.path.join(
            "..", "experiments", "graph",
            (cfg.model.name + "-" + cfg.run.name)
        )
        os.makedirs(save_dir, exist_ok=True)

        rmses_mean, rmses_std = np.mean(rmses), np.std(rmses)
        nses_mean, nses_std = np.mean(nses), np.std(nses)

        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            f.write("RMSE:\n")
            f.write(f"[{', '.join(format(x, '.3f') for x in rmses)}]\n")
            f.write("Mean, std: ")
            f.write(f"{rmses_mean:.3f}, ({rmses_std:.3f})\n\n")
            f.write("NSE:\n")
            f.write(f"[{', '.join(format(x, '.3f') for x in nses)}]\n")
            f.write("Mean, std: ")
            f.write(f"{nses_mean:.3f}, ({nses_std:.3f})")

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

    return targets, predictions


if __name__ == "__main__":
    train()
