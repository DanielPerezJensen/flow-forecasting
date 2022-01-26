# Machine learning imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
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


def train(model_name, time_features=False, plot=False):

    dataset = data.gather_river_flow_data(lags=12, time_features=time_features)
    n = len(dataset)

    train_set, val_set = torch.utils.data.random_split(
                            dataset, [round(n * 0.8), round(n * 0.2)]
                        )

    batch_size = 1
    loss_fn = nn.MSELoss(reduction="mean")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=False, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=8)

    # Crude way to distinguish between models
    # TODO add parameters to parser
    if model_name == "MLP":
        input_dim = len(train_set[0][0])

        model = models.MLP(inputs=input_dim, loss_fn=loss_fn)

    elif model_name == "GRU":

        input_dim = len(train_set[0][0])
        hidden_dim = 64
        output_dim = 1
        layer_dim = 3
        dropout_prob = 0.2
        n_epochs = 100
        learning_rate = 1e-3
        weight_decay = 1e-6

        model = models.GRU(input_dim, hidden_dim, layer_dim, output_dim,
                           dropout_prob, lr=1e-3, loss_fn=loss_fn)

    trainer = pl.Trainer(gpus=0, precision="bf16", max_epochs=5)
    trainer.fit(model, train_loader, val_loader)

    # Evaluation after training
    predictions, values = utils.evaluate(model, val_set)
    df_results = utils.format_predictions(predictions, values)
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
    parser.add_argument("--time_features", default=0, type=int,
                        help="Include time as a (cyclical) feature")
    parser.add_argument("--plot", default=0, type=int,
                        help="Plot the predictions of the validation set")

    args = parser.parse_args()

    train(args.model_name,
          time_features=args.time_features,
          plot=args.plot)
