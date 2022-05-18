# Data processing imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Standard imports
import collections
import itertools
import os
from tqdm import tqdm
import argparse

# Custom imports
import data
import utils
import plotting
from data import RiverFlowDataset

# Typing imports
from typing import Dict, Union, Tuple
import numpy.typing as npt


class PreviousMonthPredictor:
    def __init__(self, river_flow_data: RiverFlowDataset) -> None:
        self.dataset = river_flow_data
        self.fit()

    def fit(self) -> None:
        self.fit_dict = collections.OrderedDict()

        keys = self.dataset.data_date_dict.keys()

        # Pairwise iteration over the keys
        a, b = itertools.tee(keys)
        next(b, None)

        for date, next_date in zip(a, b):
            self.fit_dict[next_date] = self.dataset.get_item_by_date(date)[1].item()

    def predict(self, inp: Union[str, np.datetime64]) -> float:
        date = np.datetime64(inp)
        return self.fit_dict[date]


class AverageMonthPredictor:
    def __init__(self, river_flow_data: RiverFlowDataset) -> None:
        self.dataset = river_flow_data
        self.fit()

    def fit(self) -> None:
        self.month_data = collections.defaultdict(list)

        for date in self.dataset.data_date_dict.keys():
            date_string = date.astype(str)  # type: str
            date_string_split = date_string.split('-')
            month = int(date_string_split[1])
            self.month_data[month].append(self.dataset.get_item_by_date(date)[1].item())

        self.month_fit = {}  # type: Dict[int, float]

        for month in self.month_data:
            self.month_fit[month] = np.mean(self.month_data[month])

    def predict(self, inp: Union[str, np.datetime64]) -> float:
        date = np.datetime64(inp)
        date_string = date.astype(str)  # type: str
        date_string_split = date_string.split('-')
        month = int(date_string_split[1])
        return self.month_fit[month]


def predict_dataset(
    model: Union[AverageMonthPredictor, PreviousMonthPredictor],
    dataset: RiverFlowDataset
) -> Tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:

    inputs = []
    predictions = []
    gt = []

    for date, (inp, out) in dataset.data_date_dict.items():
        inputs.append(inp.cpu().numpy())
        predictions.append(model.predict(date))
        gt.append(out.item())

    return np.array(inputs), np.array(predictions), np.array(gt)


def baseline(args: argparse.Namespace) -> None:
    processed_path = os.path.join("data", "processed")

    dataset = data.RiverFlowDataset(
                root=processed_path,
                process=True,
                lag=6, freq="M"
            )

    # Split dataset into training, validation and test
    train, val, test = data.split_dataset(dataset, freq="M", lag=6,
                                          val_year_min=1999,
                                          val_year_max=2004,
                                          test_year_min=1974,
                                          test_year_max=1981)

    model: Union[AverageMonthPredictor, PreviousMonthPredictor]

    if args.baseline == "AverageMonth":

        model = AverageMonthPredictor(dataset)

    elif args.baseline == "PreviousMonth":

        model = PreviousMonthPredictor(dataset)

    model.fit()

    inputs, predictions, values = predict_dataset(model, test)

    index = list(test.data_date_dict.keys())

    df_results = utils.format_predictions(inputs, predictions, values, index)

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
                        help="Baseline to analyze",
                        choices=["PreviousMonth", "AverageMonth"])
    parser.add_argument("--plot", action="store_true",
                        help="Plot?")

    args = parser.parse_args()

    baseline(args)
