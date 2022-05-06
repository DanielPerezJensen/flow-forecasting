import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import os

import models
import data


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig) -> None:

    processed_path = os.path.join("data", "processed")
    dataset = data.GraphRiverFlowDataset(
                processed_path,
                scaler_name=cfg.data.scaler_name,
                freq=cfg.data.freq,
                lag=cfg.data.lag,
                lagged_variables=cfg.data.lagged_variables,
                target_variable=cfg.data.target_variable,
                target_stations=cfg.data.target_stations,
                process=True
            )

    # Split dataset into training, validation and test
    train, val, test = data.split_dataset(dataset, freq=cfg.data.freq,
                                          lag=cfg.data.lag)

    train_loader = DataLoader(train, batch_size=cfg.training.batch_size,
                              num_workers=8)
    val_loader = DataLoader(val, batch_size=cfg.training.batch_size,
                            num_workers=8)
    test_loader = DataLoader(test, batch_size=cfg.training.batch_size,
                             num_workers=8)

    # Extract optimizer
    optimizer = get_optimizer("Adam")

    # Extract some information about the graph in our dataset
    data_sample = dataset[0]
    metadata = data_sample.metadata()

    model = models.HeteroGLSTM_pl(cfg, metadata, optimizer)

    # Dummy pass to initialize all layers
    with torch.no_grad():
        model(data_sample.x_dict, data_sample.edge_index_dict)

    pl.seed_everything(42, workers=True)

    trainer = pl.Trainer(gpus=cfg.training.gpu,
                         max_epochs=cfg.training.epochs,
                         deterministic=False)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


def get_optimizer(optimizer_name: str) -> torch.optim.Optimizer:

    optimizer_dict = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW
    }

    return optimizer_dict[optimizer_name]


if __name__ == "__main__":
    train()
