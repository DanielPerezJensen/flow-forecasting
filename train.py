import torch
import pytorch_lightning as pl
from torch import nn
from torch_geometric.loader import DataLoader

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import data
import utils

import os


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:

    # Load data
    data_root = os.path.join(get_original_cwd(), "data", "processed")
    dataset = data.GraphRiverFlowDataset(data_root, **cfg.data)

    # TODO: Add date splitting to dataset class

    # Move dataset into dataloader for easy batching
    dataloader = DataLoader(dataset, cfg.training.batch_size)
    print(next(iter(dataloader)))

    # TODO: Add training code


if __name__ == "__main__":
    train()