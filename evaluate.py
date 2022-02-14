# Machine learning imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
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


def evaluate(version, checkpoint):
    path = os.path.join("lightning_logs", version,
                        "checkpoints", checkpoint + ".ckpt")

    model, checkpoint = utils.load_model(path)
    hparams = checkpoint["hyper_parameters"]
    scaler = hparams["scaler"]

    time_features = hparams["time_features"]
    index_features = hparams["index_features"]

    model.eval()

    df_features = data.gather_river_flow_data(lag=6,
                                              time_features=time_features,
                                              index_features=index_features)

    df_train, df_val, df_test = data.split_data(df_features, 6)
    _, _, X_test_arr, _, _, y_test_arr = data.scale_data(scaler, df_train, df_val, df_test)

    test_set = data.RiverFlowDataset(X_test_arr, y_test_arr)
    test_loader = DataLoader(test_set)

    predictions, values = utils.predict(model, test_loader, hparams.get("input_dim", None))

    df_results = utils.format_predictions(predictions, values, df_test, scaler=scaler)

    results_metrics = utils.calculate_metrics(df_results)

    print("Metrics of predicted values:")
    for key, val in results_metrics.items():
        print(f"{key.upper()}: {val:.3f}")

    plotting.plot_predictions(df_results)
    plotting.plot_ind_predictions(df_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--run", default="version_0", type=str,
                        help="Run Folder")
    parser.add_argument("--checkpoint", default="epoch=9-step=4459", type=str,
                        help="Checkpoint name")

    args = parser.parse_args()

    evaluate(args.run, args.checkpoint)
