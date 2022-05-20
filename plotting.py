import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


def plot_predictions(df_result: pd.DataFrame) -> None:
    """
    This function plots the predictions given in df_result
    against the actual ground truth values.
    Args:
        df_result: dataframe coming from utils.format_predictions
    """
    data = []

    value = go.Scatter(
        x=df_result.index,
        y=df_result.value,
        mode="lines",
        name="values",
        marker=dict(),
        text=df_result.index,
        line=dict(color="rgba(0,0,0, 0.3)"),
    )

    data.append(value)

    prediction = go.Scatter(
        x=df_result.index,
        y=df_result.prediction,
        mode="lines",
        line={"dash": "dot"},
        name="predictions",
        marker=dict(),
        text=df_result.index,
        opacity=0.8,
    )

    data.append(prediction)

    layout = dict(
        title="Predictions vs Actual Values for the dataset",
        xaxis=dict(title="Time", ticklen=5, zeroline=False),
        yaxis=dict(title="Value", ticklen=5, zeroline=False),
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def plot_ind_predictions(df_result: pd.DataFrame) -> None:
    """
    This function randomly samples 16 datapoints from the test set and
    plots the prediction against the value while also including the
    inputs in the plot.
    Args:
        df_result: dataframe coming from utils.format_predictions
    """
    col_list = df_result.columns
    lag_cols = [i[0] for i in
                col_list.str.findall("^river_flow_.*") if len(i) > 0][::-1]
    lag_cols.append("value")

    n = 16

    sampled = df_result.sample(n=n)

    titles = sampled.index.strftime("%m/%Y").tolist()

    fig = make_subplots(rows=4, cols=4,
                        subplot_titles=titles,
                        x_title="Months",
                        y_title="River Flow")

    row_idx, col_idx = 1, 1

    for index, row in sampled.iterrows():
        data = row[lag_cols].tolist()
        x = list(range(1, len(lag_cols) + 1))

        x_pred = [len(data) - 1, len(data)]
        y_pred = [data[-2], row["prediction"]]

        fig.add_trace(go.Scatter(
                        x=x,
                        y=row[lag_cols],
                        line={"dash": "solid"},
                    ), row_idx, col_idx)

        fig.add_trace(go.Scatter(
                        x=x_pred,
                        y=y_pred,
                        line={"dash": "dot"},
                    ), row_idx, col_idx)

        row_idx += 1

        if row_idx % 5 == 0:
            row_idx = 1
            col_idx += 1

    fig.update_layout(showlegend=False)
    fig.show()
