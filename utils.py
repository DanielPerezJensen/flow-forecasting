import torch
import torch.nn as nn
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
        df[col] = scaler.inverse_transform(df[col])

    return df


def format_predictions(
    predictions: npt.NDArray[float], values: npt.NDArray[float],
    index: List[np.datetime64], scaler: ScalerType = None
) -> pd.DataFrame:
    """
    Format predictions and values into dataframe for easy plotting
    Args:
        predictions: predicted values
        values: actual ground truth values
        df_test: the dataframe containing the data we used to evaluate
        scaler: optional arg, only used if we used a scaler during training
    """
    # vals = np.concatenate(values, axis=0).ravel()
    # preds = np.concatenate(predictions, axis=0).ravel()

    df_result = pd.DataFrame(data={"value": values, "prediction": predictions},
                             index=index)

    # merge = pd.merge(df_result, df_test, left_index=True, right_on="date")
    merge = df_result

    if scaler is not None:
        merge = inverse_transform(scaler, merge, ["value", "prediction"])

    return merge


def calculate_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates various metrics on a df containing actual targets
    and predicted targets
    Args:
        results_df: dataframe containing preds and gt
    """
    return {
        "mse": mean_squared_error(results_df.value,
                                  results_df.prediction),
        "rmse": mean_squared_error(results_df.value,
                                   results_df.prediction) ** 0.5,
        "r2": r2_score(results_df.value, results_df.prediction)
        }
