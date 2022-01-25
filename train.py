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
import datasets
import utils


def train(model_name):

    processed_folder_path = os.path.join("data", "processed")

    # Import river flow data and only preserve datapoints after 1965
    df_DGA = pd.read_csv(os.path.join(processed_folder_path, "DGA.csv"),
                         index_col=0, parse_dates=["date"])
    df_DGA = df_DGA.loc[df_DGA["date"].dt.year >= 1965]

    # Extract average monthly river flow
    monthly_flow_data_mean = df_DGA.groupby(
                                    pd.PeriodIndex(df_DGA['date'], freq="M")
                                )['river_flow'].mean()
    flow_mean_df = monthly_flow_data_mean.reset_index()

    # Convert dataset to lagged dataset
    df_generated = utils.generate_lags(flow_mean_df, "river_flow", 12)

    dataset = datasets.RiverFlowDataset(df_generated)
    n = len(dataset)

    train_set, val_set = torch.utils.data.random_split(dataset, [round(n * 0.8), round(n * 0.2)])

    batch_size = 1

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

    # Crude way to distinguish between models
    # TODO add parameters to parser
    if model_name == "MLP":
        model = models.MLP(loss_fn=nn.MSELoss(reduction="mean"))

    elif model_name == "GRU":

        input_dim = 12
        hidden_dim = 64
        output_dim = 1
        layer_dim = 3
        dropout_prob = 0.2
        n_epochs = 100
        learning_rate = 1e-3
        weight_decay = 1e-6

        model = models.GRU(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, lr=1e-3, loss_fn=nn.MSELoss(reduction="mean"))

    trainer = pl.Trainer(gpus=0, precision="bf16", max_epochs=100)
    trainer.version = "GRU"
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--model_name")

    args = parser.parse_args()

    train(args.model_name)
