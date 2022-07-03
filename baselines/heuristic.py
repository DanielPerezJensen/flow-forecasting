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
from typing import Dict, Union, Tuple, Any, List
import numpy.typing as npt
import pandas as pd


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
            next_value = self.dataset.get_item_by_date(date)[1][0][0].item()
            self.fit_dict[next_date] = next_value

    def predict(self, inp: Union[str, np.datetime64]) -> List[float]:
        date = np.datetime64(inp, "D")

        out = [self.fit_dict[date] for _ in range(6)]

        return out


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
            self.month_data[month].append(
                self.dataset.get_item_by_date(date)[1][0][0].item()
            )

        self.month_fit = {}  # type: Dict[int, float]

        for month in self.month_data:
            self.month_fit[month] = np.mean(self.month_data[month])

    def predict(self, inp: Union[str, np.datetime64]) -> float:
        date = np.datetime64(inp, "D")

        date_range = pd.date_range(date, periods=6, freq="M")

        out = [self.month_fit[int(np.datetime64(date, "D").astype(str).split("-")[1])] for date in date_range]

        return out


def predict_dataset(
    model: Union[AverageMonthPredictor, PreviousMonthPredictor],
    dataset: RiverFlowDataset
) -> Tuple[npt.NDArray[float], npt.NDArray[float], npt.NDArray[float]]:

    inputs = []
    predictions = []
    values = []

    for date, (inp, out) in dataset.data_date_dict.items():
        inputs.append(inp.squeeze().numpy())
        predictions.append(model.predict(date))
        values.append(out.item())

    return np.array(inputs), np.array(predictions), np.array(values)


def main(args: argparse.Namespace) -> None:
    processed_path = os.path.join("data", "processed")

    dataset = data.RiverFlowDataset(
                root=processed_path,
                process=True,
                freq="M"
            )

    # Split dataset into training, validation and test
    train, val, test = data.split_dataset(dataset, {},
                                          val_year_min=1999,
                                          val_year_max=2004,
                                          test_year_min=1974,
                                          test_year_max=1981)

    model: Union[AverageMonthPredictor, PreviousMonthPredictor]

    if args.model == "AverageMonth":

        model = AverageMonthPredictor(dataset)

    elif args.model == "PreviousMonth":

        model = PreviousMonthPredictor(dataset)

    model.fit()

    save_dir = os.path.join("..", "experiments", "heuristics", args.model)
    os.makedirs(save_dir, exist_ok=True)

    summer_trgs, summer_preds = gather_preds(model, True, val, test)
    trgs, preds = gather_preds(model, False, val, test)

    summer_months = np.stack((summer_trgs, summer_preds), axis=0)[None]
    all_months = np.stack((trgs, preds), axis=0)[None]

    with open(os.path.join(save_dir, "summer.npy"), "wb") as f:
        np.save(f, summer_months)

    with open(os.path.join(save_dir, "all.npy"), "wb") as f:
        np.save(f, all_months)

def gather_preds(
    model: Union[AverageMonthPredictor, PreviousMonthPredictor],
    summer: bool, *datasets: data.RiverFlowDataset
) -> Any:

    new_dataset = data.RiverFlowDataset()

    for dataset in datasets:
        for date, value in dataset.data_date_dict.items():

            if summer:
                # Third month is split between winter and summer
                if int(date.astype(str).split("-")[1]) == 8:
                    new_dataset.set_data(date, value)
            else:
                new_dataset.set_data(date, value)

    targets, predictions = [], []

    for date in new_dataset.data_date_dict:

        _, target = new_dataset.get_item_by_date(date)
        out = model.predict(date)

        targets.append(target.cpu().numpy())
        predictions.append(np.array(out).reshape((-1, 6)))

    targets = np.array(targets)
    predictions = np.array(predictions)

    return targets, predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument("--model", default="PreviousMonth", type=str,
                        help="Baseline to analyze",
                        choices=["PreviousMonth", "AverageMonth"])

    args = parser.parse_args()

    main(args)
