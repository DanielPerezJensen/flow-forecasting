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


class PreviousMonthPredictor:
    def __init__(self, river_flow_data):
        self.river_flow_data = river_flow_data

    def fit(self):
        self.date_fit = self.river_flow_data.copy()
        self.date_fit[f"river_flow_1"] = self.river_flow_data["river_flow"].shift(1)
        self.date_fit = self.date_fit.iloc[1:]

        self.date_fit.date = self.date_fit.date.dt.to_timestamp().dt.strftime("%Y-%m")

    def predict(self, inp):
        date = inp.date.to_timestamp().strftime("%Y-%m")
        return self.date_fit.loc[self.date_fit.date == date]["river_flow_1"].iloc[0]


class MonthPredictor:
    def __init__(self, river_flow_data):
        self.river_flow_data = river_flow_data

    def fit(self):
        self.river_flow_data["month"] = self.river_flow_data["date"].dt.month
        monthly_data = self.river_flow_data.groupby("month")["river_flow"].mean()
        self.month_fit = monthly_data.to_dict()

    def predict(self, inp):
        month = int(inp.date.to_timestamp().strftime("%m"))
        return self.month_fit[month]


def predict_testset(model, df_test):
    predictions = []
    values = []

    for index, row in df_test.iterrows():
        predictions.append([model.predict(row)])
        values.append([row.river_flow])

    return np.array(predictions), np.array(values)


def baseline(args):

    df_features = data.gather_river_flow_data(lag=6)
    df_train, df_val, df_test = data.split_data(df_features, 0)

    if args.baseline == "Month":

        model = MonthPredictor(df_train)

    elif args.baseline == "PreviousMonth":

        model = PreviousMonthPredictor(df_features)

    model.fit()

    predictions, values = predict_testset(model, df_test)

    df_results = utils.format_predictions(predictions, values, df_test)

    results_metrics = utils.calculate_metrics(df_results)

    print("Metrics of predicted values:")
    for key, val in results_metrics.items():
        print(f"{key.upper()}: {val:.3f}")

    if args.plot:
        plotting.plot_predictions(df_results)
        plotting.plot_ind_predictions(df_results)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--baseline", default="PreviousMonth", type=str,
                        help="Baseline to analyze")
    parser.add_argument("--plot", action="store_true",
                        help="Plot?")

    args = parser.parse_args()

    baseline(args)
