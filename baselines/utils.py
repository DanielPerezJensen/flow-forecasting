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


def inverse_transform(
    scaler: ScalerType, df: pd.DataFrame, columns: List[str]
) -> pd.DataFrame:
    """
    Transforms values in df columns back to normal values using sklearn scaler
    """
    for col in columns:
        df[col] = scaler.inverse_transform(df[col].to_numpy().reshape(-1, 1))

    return df


def predict(
    model: pl.LightningModule, test_loader: DataLoader
) -> Tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:
    """
    This function gathers the predictions from the model and the
    actual values found in the test_loader.
    Args:
        model: PytorchLightningModule
        test_loader: pytorch.DataLoader
    """
    model.eval()

    inputs = []
    predictions = []
    values = []

    for i, data in enumerate(test_loader):
        inp, targets = data

        if model.cfg.model.name in ["GRU", "LSTM"]:
            inp = inp.view([1, -1, model.input_dim])

        inp = inp.to(model.device)
        targets = targets.to(model.device)

        outputs = model(inp)

        inputs.append(inp.detach().cpu().squeeze().numpy()[:, 0])
        predictions.append(outputs.detach().cpu().numpy().reshape((model.cfg.data.n_preds)))
        values.append(targets.detach().cpu().numpy().reshape((model.cfg.data.n_preds)))

    return np.array(inputs), np.array(predictions), np.array(values)


def format_predictions(
    inputs: npt.NDArray[float], predictions: npt.NDArray[float],
    values: npt.NDArray[float], index: List[np.datetime64],
    scaler: ScalerType = None
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

    prediction_cols = [f"prediction_{i + 1}" for i in range(len(predictions[0]))]
    predictions = predictions.T
    df_result = df_result.assign(**dict(zip(prediction_cols, predictions)))

    value_cols = [f"river_flow+{i + 1}" for i in range(len(values[0]))]
    values = values.T
    df_result = df_result.assign(**dict(zip(value_cols, values)))

    columns = input_cols + prediction_cols + value_cols

    if scaler is not None:
        df_result = inverse_transform(scaler, df_result, columns)

    return df_result
