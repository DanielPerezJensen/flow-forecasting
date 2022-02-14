import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import models


def predict(model, test_loader, input_dim=None):
    """
    This function gathers the predictions from the model and the
    actual values found in the test_loader. Input_dim is only
    a value if the inputted model is the GRU.
    Args:
        model: PytorchLightningModule
        test_loader: pytorch.DataLoader
        input_dim: optional arg, only used if model is a GRU
    """
    model.eval()
    total_loss = 0.0

    predictions = []
    values = []

    for i, data in enumerate(test_loader):
        inputs, targets = data
        if input_dim:
            inputs = inputs.view([1, -1, input_dim])

        inputs = inputs.to(model.device)
        targets = targets.to(model.device)

        outputs = model(inputs)

        predictions.append(outputs.detach().cpu().numpy())
        values.append(targets.detach().cpu().numpy())

    return predictions, values


def inverse_transform(scaler, df, columns):
    """
    Transforms values in df columns back to normal values using sklearn scaler
    """
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])

    return df


def format_predictions(predictions, values, df_test, scaler=None):
    """
    Format predictions and values into dataframe for easy plotting
    Args:
        predictions: predicted values
        values: actual ground truth values
        df_test: the dataframe containing the data we used to evaluate
        scaler: optional arg, only used if we used a scaler during training
    """
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()

    df_result = pd.DataFrame(data={"value": vals, "prediction": preds},
                             index=df_test.head(len(vals)).date)

    merge = pd.merge(df_result, df_test, left_index=True, right_on="date")
    merge = merge.set_index("date")
    merge.index = merge.index.to_timestamp()

    if scaler is not None:
        merge = inverse_transform(scaler, merge, [["value", "prediction"]])

    return merge


def calculate_metrics(results_df):
    """
    Calculates various metrics on a df containing actual targets
    and predicted targets
    Args:
        results_df: dataframe containing preds and gt
    """
    return {'mse': mean_squared_error(results_df.value,
                                      results_df.prediction),
            'rmse': mean_squared_error(results_df.value,
                                       results_df.prediction) ** 0.5,
            'r2': r2_score(results_df.value, results_df.prediction)}


def load_model(ckpt_path):
    """
    Loads a model from the given ckpt_path, keep in mind that the ckpt_path
    must be a stored model of the same module.
    Args:
        model: pl.LightningModule
        ckpt_path: str, denoting path to model
    """
    checkpoint = torch.load(ckpt_path)
    hparams = checkpoint["hyper_parameters"]
    model_name = hparams["name"]

    print(hparams)

    if model_name == "MLP":
        model = models.MLP(**hparams)
    elif model_name == "GRU":
        model = models.GRU(**hparams)

    model = model.load_from_checkpoint(ckpt_path)

    return model, checkpoint
