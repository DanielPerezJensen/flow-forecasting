# Machine learning imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import wandb
import sklearn

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
import utils
import plotting


def train(args):

    run_metrics = collections.defaultdict(list)

    for seed in args.seeds:

        df_features = data.gather_data(
                        lag=args.lag,
                        time_features=args.time_features,
                        index_features=args.index_features,
                        index_surf_features=args.index_surf_features,
                        index_cloud_features=args.index_cloud_features
                    )

        pl.seed_everything(seed, workers=True)

        scaler = data.get_scaler(args.scaler)

        # Split data and convert it to a torch.Dataset
        df_train, df_val, df_test = data.split_data(df_features, args.lag)
        arrays = data.scale_data(scaler, df_train, df_val, df_test)
        X_train, X_val, X_test, y_train, y_val, y_test = arrays

        train_set = data.RiverFlowDataset(X_train, y_train)
        val_set = data.RiverFlowDataset(X_val, y_val)
        test_set = data.RiverFlowDataset(X_test, y_test)

        # Convert Datasets to Dataloader to allow for training
        batch_size = args.batch_size

        train_loader = DataLoader(train_set, batch_size=batch_size,
                                  shuffle=False, num_workers=8,)
        test_loader = DataLoader(test_set, batch_size=1,
                                 shuffle=False, num_workers=8)
        val_loader = DataLoader(val_set, batch_size=batch_size,
                                shuffle=False, num_workers=8)

        # Some general parameters that are valid for all models
        model_name = args.model_name
        input_dim = len(train_set[0][0])
        output_dim = 1

        # Store some default params in config
        config = vars(args)

        config["input_dim"] = len(train_set[0][0])
        config["output_dim"] = 1
        config["scaler"] = scaler

        model = utils.get_model(config)

        wandb_logger = WandbLogger(project="test", entity="danielperezjensen",
                                   offline=args.wandb)

        trainer = pl.Trainer(gpus=int(args.gpu), log_every_n_steps=10,
                             max_epochs=args.epochs, logger=wandb_logger)
        trainer.fit(model, train_loader, test_loader)

        wandb.finish(quiet=True)

        # Evaluation after training
        predictions, values = utils.predict(model, test_loader)

        df_results = utils.format_predictions(predictions, values, df_test,
                                              scaler=scaler)

        results_metrics = utils.calculate_metrics(df_results)

        print("Metrics of predicted values:")
        for key, val in results_metrics.items():
            run_metrics[key].append(val)
            print(f"{key.upper()}: {val:.3f}")

        if args.plot:
            plotting.plot_predictions(df_results)
            plotting.plot_ind_predictions(df_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--model_name", default="MLP", type=str,
                        help="Model to be trained",
                        choices=["GRU", "MLP", "LSTM"])
    parser.add_argument("--seeds", default=[52], type=int,
                        nargs="+", help="Set seeds of runs")
    parser.add_argument("--param_set", default=1, type=int,
                        help="Choose param set from models.json")

    parser.add_argument("--lag", default=6, type=int,
                        help="time lag to use as features")

    parser.add_argument("--batch_size", default=16, type=int,
                        help="Size of batches during training")
    parser.add_argument("--epochs", default=50, type=int,
                        help="Number of epochs to train model for")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument("--weight_decay", default=1e-6, type=float,
                        help="Weight decay")
    parser.add_argument("--scaler", default="maxabs", type=str,
                        help="Scaler to use for the values",
                        choices=["none", "minmax", "standard",
                                 "maxabs", "robust"])

    parser.add_argument("--time_features", default=0, choices=[0, 1], type=int,
                        help="Include time as a (cyclical) feature")
    parser.add_argument("--index_features", default=0,
                        choices=[0, 1], type=int,
                        help="Include NDSI/NDVI as a feature")
    parser.add_argument("--index_surf_features", default=0,
                        choices=[0, 1], type=int,
                        help="Include NDSI/NDVI area as a feature")
    parser.add_argument("--index_cloud_features", default=0,
                        choices=[0, 1], type=int,
                        help="Include NDSI/NDVI cloud cover as a feature")

    parser.add_argument("--save_run", action="store_true",
                        help="Save the metrics of this run to run_metrics.csv")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the predictions of the validation set")
    parser.add_argument("--wandb", action="store_false",
                        help="Flag to set if you want to upload to wandb.ai")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training")

    args = parser.parse_args()

    train(args)
