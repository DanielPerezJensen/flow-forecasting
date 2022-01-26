import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(model, validation_set, input_dim=None):
    model.eval()
    total_loss = 0.0

    predictions = []
    values = []

    testloader = DataLoader(validation_set, batch_size=1, num_workers=2)

    for i, data in enumerate(tqdm(testloader)):
        inputs, targets = data
        if input_dim:
            inputs = inputs.view([1, -1, input_dim])

        inputs = inputs.to(model.device)
        targets = targets.to(model.device)

        outputs = model(inputs)

        predictions.append(outputs.detach().cpu().numpy())
        values.append(targets.detach().cpu().numpy())

    return predictions, values


def format_predictions(predictions, values, scaler=None):
    """
    Format predictions and values into dataframe for easy plotting
    """
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()

    df_result = pd.DataFrame(data={"value": vals, "prediction": preds})

    return df_result


def calculate_metrics(df):
    """
    Calculates various metrics on a df containing actual targets
    and predicted targets
    """
    return {'mse': mean_squared_error(df.value, df.prediction),
            'rmse': mean_squared_error(df.value, df.prediction) ** 0.5,
            'r2': r2_score(df.value, df.prediction)}
