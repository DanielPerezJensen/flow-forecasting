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

    processed_path = os.path.join(get_original_cwd(), "data", "processed")
    dataset = data.GraphFlowDataset(
                root=processed_path,
                graph_type=cfg.data.graph_type,
                scaler_name=cfg.data.scaler_name,
                freq=cfg.data.freq,
                lag=cfg.data.lag,
                lagged_vars=cfg.data.lagged_variables,
                target_var=cfg.data.target_variable,
                target_stations=cfg.data.target_stations,
                process=True
            )

    # Split dataset into training, validation and test
    train, val, test = data.split_dataset(dataset, freq=cfg.data.freq,
                                          lag=cfg.data.lag,
                                          val_year_max=2006,
                                          test_year_max=2016)

    train_loader = DataLoader(train, batch_size=cfg.training.batch_size,
                              num_workers=8, shuffle=True)
    val_loader = DataLoader(val, batch_size=cfg.training.batch_size,
                            num_workers=8)
    test_loader = DataLoader(test, batch_size=cfg.training.batch_size,
                             num_workers=8)

    # Extract some information about the graph in our dataset
    data_sample = dataset[0]
    metadata = data_sample.metadata()

    # scaler = train.scaler
    scaler = dataset.scaler
    model = models.HeteroGLSTM_pl(cfg, metadata, scaler)

    # Dummy pass to initialize all layers
    with torch.no_grad():
        model(data_sample.x_dict, data_sample.edge_index_dict)

    pl.seed_everything(42, workers=True)

    trainer = pl.Trainer(gpus=cfg.training.gpu,
                         max_epochs=cfg.training.epochs,
                         deterministic=False)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    train()
