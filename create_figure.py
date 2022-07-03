import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
import argparse

from typing import Any, List


plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '100'})


def main(
    root: str, summer: bool, files: List[str], labels: List[str],
    mean_all: bool, stations: bool, scatter: bool
) -> None:

    if mean_all:
        plot_all(root, summer, files, labels)

    if stations:
        plot_stations(root, summer, files, labels)

    if scatter:
        plot_scatter(root, summer, files, labels)


def plot_all(
    root: str, summer: bool, files: List[str], labels: List[str]
) -> None:

    for file, label in zip(files, labels):
        file_dir = os.path.join(root, file)

        if summer:
            title = "RMSE for prediction of summer months"
            filename = "summer.npy"
            months = [10, 11, 12, 1, 2, 3]
            x = [datetime(1, int(m), 1).strftime("%b") for m in months]
        else:
            title = "RMSE for prediction of all months in dataset"
            filename = "all.npy"
            x = np.arange(6)

        with open(os.path.join(file_dir, filename), "rb") as f:
            months = np.load(f)

        if months.shape[0] == 1:
            capsize = 0
        else:
            capsize = 3

        targets = months[:, 0]
        predictions = months[:, 1]

        mse = (targets - predictions) ** 2

        mean_mse = mse.mean(axis=(1, 2))

        mean_rmse = np.sqrt(mean_mse).mean(axis=0)
        sqrt_rmse = np.sqrt(mean_mse).std(axis=0)

        plt.errorbar(
            x, mean_rmse, yerr=sqrt_rmse / 2, capsize=capsize,
            label=f"{label}-$\\mu={np.mean(mean_rmse):.3f}$"
        )

    plt.title(title)
    plt.xlabel("$\\text{n\\textsuperscript{th}}$ prediction")
    plt.ylabel("RMSE")

    plt.legend()
    plt.show()


def plot_stations(
    root: str, summer: bool, files: List[str], labels: List[str]
) -> None:
    for file, label in zip(files, labels):
        file_dir = os.path.join(root, file)

        if summer:
            title = "RMSE for prediction of summer months"
            filename = "summer.npy"
            # plt.figure(figsize=(10, 6))
            months = [10, 11, 12, 1, 2, 3]
            x = [datetime(1, int(m), 1).strftime("%b") for m in months]
        else:
            title = "RMSE for prediction of all months in dataset"
            filename = "all.npy"
            x = np.arange(6)

        with open(os.path.join(file_dir, filename), "rb") as f:
            months = np.load(f)

        targets = months[:, 0]
        predictions = months[:, 1]

        if months.shape[0] == 1:
            capsize = 0
        else:
            capsize = 3

        station_dict = {
            0: "Pastillo", 1: "Jorquera", 2: "Pulido", 3: "Manflas"
        }

        for station in range(targets.shape[2]):

            station_targets = targets[:, :, station, :]
            station_predictions = predictions[:, :, station, :]

            mse = (station_targets - station_predictions) ** 2

            mean_mse = mse.mean(axis=(1))
            mean_rmse = np.sqrt(mean_mse).mean(axis=0)
            sqrt_rmse = np.sqrt(mean_mse).std(axis=0)

            plt.figure(station)

            plt.errorbar(
                x, mean_rmse, yerr=sqrt_rmse / 2, capsize=capsize,
                label=f"{label}-$\\mu={np.mean(mean_rmse):.3f}$"
            )

            plt.title(title + ": " + station_dict[station])
            plt.xlabel("$\\text{n\\textsuperscript{th}}$ prediction")
            plt.ylabel("RMSE")
            plt.legend()

    plt.show()


def plot_scatter(
    root: str, summer: bool, files: List[str], labels: List[str]
) -> None:
    figure_count = 0

    for i, (file, label) in enumerate(zip(files, labels)):
        file_dir = os.path.join(root, file)

        station_dict = {
            0: "Pastillo", 1: "Jorquera", 2: "Pulido", 3: "Manflas"
        }

        color_dict = {
            0: "tab:red", 1: "tab:blue", 2: "tab:purple", 3: "tab:green"
        }

        if summer:
            title = "Scatter of predictions vs targets of summer months"
            filename = "summer.npy"
            # plt.figure(figsize=(10, 6))
            months = [10, 11, 12, 1, 2, 3]
            x = [datetime(1, int(m), 1).strftime("%b") for m in months]
        else:
            title = "Scatter of predictions vs targets of all months in dataset"
            filename = "all.npy"
            x = np.arange(6)

        with open(os.path.join(file_dir, filename), "rb") as f:
            months = np.load(f)

        targets = months[:, 0].mean(axis=0)
        predictions = months[:, 1].mean(axis=0)

        for lag in range(targets.shape[2]):

            plt.figure(figure_count)
            figure_count += 1

            biggest_target = np.max(
                [int(np.ceil(np.max(predictions[:, :, lag]))),
                 int(np.ceil(np.max(targets[:, :, lag])))]
            )

            # Plot 1-1 trendline
            plt.plot([i for i in range(biggest_target + 1)], color="k",
                     linestyle=":", label="(1,1) line")

            for station in range(targets.shape[1]):

                lag_targets = targets[:, station, lag]
                lag_predictions = predictions[:, station, lag]

                z = np.polyfit(lag_predictions, lag_targets, 1)

                new_x = range(int(np.ceil(np.max(predictions[:, :, lag]))))
                y_hat = np.poly1d(z)(new_x)

                plt.scatter(
                    lag_predictions, lag_targets,
                    label=station_dict[station], color=color_dict[station],
                    marker="x"
                )

                plt.plot(new_x, y_hat,
                         color=color_dict[station],
                         linestyle="--", alpha=0.6)

                plt.title(label + "-" + str(lag) + "-" + title)
                plt.xlabel("Prediction")
                plt.ylabel("Target")
                plt.legend()

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--plot_all", action="store_true")
    parser.add_argument("--plot_stations", action="store_true")
    parser.add_argument("--plot_scatter", action="store_true")
    parser.add_argument("--summer", action="store_true")
    parser.add_argument('--files', metavar='N', type=str, nargs='+',
                        help='Filenames')
    parser.add_argument('--labels', metavar='N', type=str, nargs='+',
                        help='Labels')
    parser.add_argument

    args = parser.parse_args()

    main(
        "experiments/", args.summer, args.files, args.labels,
        args.plot_all, args.plot_stations, args.plot_scatter
    )
