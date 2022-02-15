# Machine learning imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
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
        df_features = data.gather_river_flow_data(
                        lag=args.lag,
                        time_features=args.time_features,
                        index_features=args.index_features,
                        index_area_features=args.index_area_features,
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
        val_loader = DataLoader(val_set, batch_size=1,
                                shuffle=False, num_workers=8)

        # Some general parameters that are valid for all models
        model_name = args.model_name
        input_dim = len(train_set[0][0])
        n_epochs = args.epochs
        learning_rate = args.lr
        loss_fn = nn.MSELoss(reduction="mean")

        if model_name == "MLP":

            model = models.MLP(inputs=input_dim, outputs=1, lr=learning_rate,
                               loss_fn=loss_fn, lag=args.lag,
                               scaler=scaler,
                               time_features=args.time_features,
                               index_features=args.index_features,
                               index_area_features=args.index_area_features,
                               index_cloud_features=args.index_cloud_features)

        elif model_name == "GRU":

            hidden_dim = 64
            output_dim = 1
            layer_dim = 3
            dropout_prob = 0.2
            weight_decay = 1e-6

            model = models.GRU(input_dim, hidden_dim, layer_dim, output_dim,
                               dropout_prob, lr=learning_rate, loss_fn=loss_fn,
                               batch_size=batch_size, scaler=scaler,
                               time_features=args.time_features,
                               index_features=args.index_features,
                               index_area_features=args.index_area_features,
                               index_cloud_features=args.index_cloud_features)

        trainer = pl.Trainer(gpus=int(args.gpu), precision="bf16",
                             max_epochs=n_epochs, log_every_n_steps=10)
        trainer.fit(model, train_loader, val_loader)

        # Evaluation after training
        if model_name == "GRU":
            predictions, values = utils.predict(model, test_loader,
                                                input_dim=input_dim)
        else:
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

    if args.save_run:
        run_data = {
            "model_type": args.model_name,
            "seeds": args.seeds,
            "lag": args.lag,
            "lr": args.lr,
            "time_features": args.time_features,
            "index_features": args.index_features,
            "index_area_features": args.index_area_features,
            "index_cloud_features": args.index_cloud_features,
            "epochs": args.epochs,
            "scaler": args.scaler,
            "mse_mean": np.mean(run_metrics["mse"]),
            "mse_std": np.std(run_metrics["mse"]),
            "rmse_mean": np.mean(run_metrics["rmse"]),
            "rmse_std": np.std(run_metrics["rmse"]),
            "r2_mean": np.mean(run_metrics["r2"]),
            "r2_std": np.std(run_metrics["r2"])
        }

        df = pd.DataFrame([run_data])

        if os.path.isfile("run_metrics.csv"):
            df_add = pd.read_csv("run_metrics.csv", index_col=0)
            df = pd.concat([df_add, df], ignore_index=True)

            df.to_csv("run_metrics.csv", float_format='%.3f')
        else:
            df.to_csv("run_metrics.csv", float_format='%.3f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--model_name", default="MLP", type=str,
                        help="Model to be trained")
    parser.add_argument("--seeds", default=[52, 86, 91, 10, 73], type=int,
                        help="Set seeds of runs")

    parser.add_argument("--lag", default=6, type=int,
                        help="time lag to use as features")

    parser.add_argument("--batch_size", default=1, type=int,
                        help="Size of batches during training")
    parser.add_argument("--epochs", default=50, type=int,
                        help="Number of epochs to train model for")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument("--scaler", default="standard", type=str,
                        help="Scaler to use for the values",
                        choices=["none", "minmax", "standard",
                                 "maxabs", "robust"])

    parser.add_argument("--time_features", action='store_true',
                        help="Include time as a (cyclical) feature")
    parser.add_argument("--index_features", action="store_true",
                        help="Include NDSI/NDVI as a feature")
    parser.add_argument("--index_area_features", action="store_true",
                        help="Include NDSI/NDVI area as a feature")
    parser.add_argument("--index_cloud_features", action="store_true",
                        help="Include NDSI/NDVI cloud cover as a feature")

    parser.add_argument("--save_run", action="store_true",
                        help="Save the metrics of this run to run_metrics.csv")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the predictions of the validation set")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training")

    args = parser.parse_args()

    train(args)
