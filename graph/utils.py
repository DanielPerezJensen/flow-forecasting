import torch
import torch.nn as nn
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer
import json
import os

from typing import Tuple, List, Union, Dict
import numpy.typing as npt

ScalerType = Union[MinMaxScaler, StandardScaler,
                   MaxAbsScaler, RobustScaler, FunctionTransformer]


def predict(
    model: pl.LightningModule, test_loader: DataLoader
) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    This function gathers the predictions from the model and the
    actual values found in the test_loader.
    Args:
        model: PytorchLightningModule
        test_loader: pytorch.DataLoader
    """
    model.eval()

    scaler = model.scaler

    predictions = []
    values = []

    for i, data in enumerate(test_loader):

        inp = data.to(model.device)
        targets = inp["measurement"].y

        if model.cfg.data.sequential:
            outputs = model(inp.xs_dict, inp.edge_indices_dict)
        else:
            outputs = model(inp.x_dict, inp.edge_index_dict)
            outputs = outputs[None, :, :]
            targets = targets[None, :, :]

        predictions.append(
            scaler.inverse_transform(outputs.detach().cpu().numpy().reshape((-1, targets.size(2))))
        )
        values.append(
            scaler.inverse_transform(targets.detach().cpu().numpy().reshape((-1, targets.size(2))))
        )

    predictions = np.array(predictions)
    values = np.array(values)

    return values, predictions


def format_predictions(
    inputs: npt.NDArray[float], predictions: npt.NDArray[float],
    values: npt.NDArray[float], index: List[np.datetime64]
) -> pd.DataFrame:
    """
    Format predictions and values into dataframe for easy plotting
    Args:
        predictions: predicted values
        values: actual ground truth values
        df_test: the dataframe containing the data we used to evaluate
        scaler: optional arg, only used if we used a scaler during training
    """

    df_result = pd.DataFrame(index=index)

    # Add previous values as lagged values
    input_cols = [f"river_flow-{i + 1}" for i in range(len(inputs[0]))]
    inputs = inputs.T
    df_result = df_result.assign(**dict(zip(input_cols, inputs)))

    prediction_cols = [
        f"prediction_{i + 1}" for i in range(len(predictions[0]))
    ]
    predictions = predictions.T
    df_result = df_result.assign(**dict(zip(prediction_cols, predictions)))

    value_cols = [f"river_flow+{i + 1}" for i in range(len(values[0]))]
    values = values.T
    df_result = df_result.assign(**dict(zip(value_cols, values)))

    columns = input_cols + prediction_cols + value_cols

    return df_result
