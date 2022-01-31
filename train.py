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
import pandas
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

    df_features = data.gather_river_flow_data(lags=args.lag, time_features=args.time_features)
    df_train, df_val, df_test = data.split_data(df_features, args.lag)

    train_set = data.RiverFlowDataset(df_train)
    test_set = data.RiverFlowDataset(df_test)
    val_set = data.RiverFlowDataset(df_val)

    # Some general parameters that are valid for all models
    model_name = args.model_name
    input_dim = len(train_set[0][0])
    batch_size = args.batch_size
    n_epochs = args.epochs
    learning_rate = args.lr
    loss_fn = nn.MSELoss(reduction="mean")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=1,
                            shuffle=False, num_workers=8)

    if model_name == "MLP":

        model = models.MLP(inputs=input_dim, loss_fn=loss_fn, lr=learning_rate)

    elif model_name == "GRU":

        hidden_dim = 64
        output_dim = 1
        layer_dim = 3
        dropout_prob = 0.2
        weight_decay = 1e-6

        model = models.GRU(input_dim, hidden_dim, layer_dim, output_dim,
                           dropout_prob, lr=learning_rate, loss_fn=loss_fn,
                           batch_size=batch_size)

    trainer = pl.Trainer(gpus=int(args.gpu), precision="bf16",
                         max_epochs=n_epochs, log_every_n_steps=10)
    trainer.fit(model, train_loader, val_loader)

    # Evaluation after training
    if model_name == "GRU":
        predictions, values = utils.evaluate(model, test_loader,
                                             input_dim=input_dim)
    else:
        predictions, values = utils.evaluate(model, test_loader)
    df_results = utils.format_predictions(predictions, values, df_test)
    results_metrics = utils.calculate_metrics(df_results)

    print("Metrics of predicted values:")
    for key, val in results_metrics.items():
        print(f"{key.upper()}: {val:.3f}")

    if plotting:
        plotting.plot_predictions(df_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--model_name", default="MLP", type=str,
                        help="Model to be trained")
    parser.add_argument("--lag", default=6, type=int,
                        help="time lag to use as features")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Size of batches during training")
    parser.add_argument("--epochs", default=50, type=int,
                        help="Number of epochs to train model for")
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument("--time_features", action='store_true',
                        help="Include time as a (cyclical) feature")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the predictions of the validation set")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for training")

    args = parser.parse_args()

    train(args)
