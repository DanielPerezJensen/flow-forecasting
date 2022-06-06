import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


def plot_ind_predictions(df_result: pd.DataFrame) -> None:
    """
    This function randomly samples 16 datapoints from the test set and
    plots the prediction against the value while also including the
    inputs in the plot.
    Args:
        df_result: dataframe coming from utils.format_predictions
    """
    df_result.index = pd.to_datetime(df_result.index)

    lagged_cols = df_result.columns[df_result.columns.str.match("river_flow-\\d")]
    target_cols = df_result.columns[df_result.columns.str.match("river_flow\\+\\d")]
    prediction_cols = df_result.columns[df_result.columns.str.match("prediction_\\d")]

    n = 16

    sampled = df_result.sample(n=n)

    titles = sampled.index.strftime("%m/%Y").tolist()

    fig = make_subplots(rows=4, cols=4,
                        subplot_titles=titles,
                        x_title="Months",
                        y_title="River Flow")

    row_idx, col_idx = 1, 1

    for index, row in sampled.iterrows():
        lagged = row[lagged_cols].tolist()
        targets = row[target_cols].tolist()
        predictions = row[prediction_cols].tolist()

        x = list(range(1, len(lagged) + 1))
        x_after = [x[-1]] + list(range(x[-1] + 1, x[-1] + 1 + len(lagged)))

        targets = [lagged[-1]] + targets
        predictions = [lagged[-1]] + predictions

        fig.add_trace(go.Scatter(
                        x=x,
                        y=lagged,
                        line={"dash": "solid", "color": "orange"},
                    ), row_idx, col_idx)

        fig.add_trace(go.Scatter(
                        x=x_after,
                        y=targets,
                        line={"dash": "dot", "color": "green"},
                    ), row_idx, col_idx)

        fig.add_trace(go.Scatter(
                        x=x_after,
                        y=predictions,

                        line={"dash": "dot", "color": "blue"},
                    ), row_idx, col_idx)

        row_idx += 1

        if row_idx % 5 == 0:
            row_idx = 1
            col_idx += 1

    fig.update_layout(showlegend=False)
    fig.show()
