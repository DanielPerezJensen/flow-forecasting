import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import models


def predict(model, test_loader):
    """
    This function gathers the predictions from the model and the
    actual values found in the test_loader. Input_dim is only
    a value if the inputted model is the GRU.
    Args:
        model: PytorchLightningModule
        test_loader: pytorch.DataLoader
    """
    model.eval()
    total_loss = 0.0

    predictions = []
    values = []

    for i, data in enumerate(test_loader):
        inputs, targets = data
        if model.name == "GRU":
            inputs = inputs.view([1, -1, model.input_dim])

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


def load_params(model_name, param_set):
    """
    Loads a parameter set from models.json
    """
    assert os.path.exists("models.json")

    with open("models.json", "r") as f:
        data = json.load(f)

    return data[model_name][str(param_set)]


def get_model(args):

    params = load_params(args.model_name, args.param_set)

    # Load default values from args
    input_dim = args.input_dim
    output_dim = args.output_dim
    loss_fn = torch.nn.MSELoss(reduction="mean")

    if args.model_name == "MLP":
        layers = params["layers"]
        weight_decay = params["weight_decay"]

        model = models.MLP(layers, inputs=input_dim, outputs=1,
                           lr=args.lr, weight_decay=weight_decay,
                           loss_fn=loss_fn, lag=args.lag, 
                           scaler=args.scaler_func,
                           time_features=args.time_features,
                           index_features=args.index_features,
                           index_area_features=args.index_area_features,
                           index_cloud_features=args.index_cloud_features)

    elif args.model_name == "GRU":
        input_dim = args.input_dim
        hidden_dim = params["hidden_dim"]
        layer_dim = params["layer_dim"]
        dropout_prob = params["dropout_prob"]
        weight_decay = params["weight_decay"]

        model = models.GRU(input_dim, hidden_dim, layer_dim, output_dim,
                           dropout_prob, lr=args.lr, loss_fn=loss_fn,
                           batch_size=args.batch_size,
                           weight_decay=weight_decay,
                           scaler=args.scaler_func,
                           time_features=args.time_features,
                           index_features=args.index_features,
                           index_area_features=args.index_area_features,
                           index_cloud_features=args.index_cloud_features)

    return model
