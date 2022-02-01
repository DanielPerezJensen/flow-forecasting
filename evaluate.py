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


def load_model(model, ckpt_path):
    model = model.load_from_checkpoint(ckpt_path)

    return model


if __name__ == "__main__":
    path = os.path.join("lightning_logs", "version_0", "checkpoints", "epoch=9-step=4459.ckpt")
    model = load_model(models.MLP, path)
    model.eval()

    df_features = data.gather_river_flow_data(lags=6, time_features=True)

    _, _, df_test = data.split_data(df_features, 6)
    test_set = data.RiverFlowDataset(df_test)
    test_loader = DataLoader(test_set)

    predictions, values = utils.predict(model, test_loader)

    df_result = utils.format_predictions(predictions, values, df_test)
    plotting.plot_ind_predictions(df_result)
