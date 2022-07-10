import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
import argparse

from typing import Any, List


plt.style.use(['science', 'ieee', 'std-colors'])
plt.rcParams.update({'figure.dpi': '100', 'figure.figsize': (6, 4)})


def main(
    root: str, summer: bool, files: List[str], labels: List[str],
    mean_all: bool, stations: bool, scatter: bool, save_dir: str
) -> None:

    if mean_all:
        if save_dir:
            os.makedirs(os.path.join(f"figures/{save_dir}/all/"), exist_ok=True)
        plot_all(root, summer, files, labels, save_dir)

    if stations:
        if save_dir:
            os.makedirs(os.path.join(f"figures/{save_dir}/stations/"), exist_ok=True)
        plot_stations(root, summer, files, labels, save_dir)

    if scatter:
        if save_dir:
            os.makedirs(os.path.join(f"figures/{save_dir}/scatter/"), exist_ok=True)
        plot_scatter(root, summer, files, labels, save_dir)


def plot_all(
    root: str, summer: bool, files: List[str], labels: List[str], save_dir: str
) -> None:

    for file, label in zip(files, labels):
        file_dir = os.path.join(root, file)

        if summer:
            title = "RMSE Curve - Summer Predictions"
            filename = "summer.npy"
            months = [10, 11, 12, 1, 2, 3]
            x = [datetime(1, int(m), 1).strftime("%b") for m in months]
        else:
            title = "RMSE Curve - All Predictions"
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

        plt.plot(
            x, mean_rmse, label=f"{label}: $\\mu={np.mean(mean_rmse):.3f}$"
        )
        plt.fill_between(x, mean_rmse - sqrt_rmse, mean_rmse + sqrt_rmse, alpha=0.2)

    plt.title(title)
    plt.xlabel("$\\text{t}^{th}$ prediction")
    plt.ylabel("RMSE")

    plt.legend()

    if save_dir:
        plt.savefig(os.path.join(f"figures/{save_dir}/all/figure.pdf"))
        plt.clf()
    else:
        plt.show()


def plot_stations(
    root: str, summer: bool, files: List[str], labels: List[str], save_dir: str
) -> None:

    for file, label in zip(files, labels):
        file_dir = os.path.join(root, file)

        if summer:
            title = "RMSE Curve - Summer Predictions"
            filename = "summer.npy"
            months = [10, 11, 12, 1, 2, 3]
            x = [datetime(1, int(m), 1).strftime("%b") for m in months]
        else:
            title = "RMSE Curve - Summer Predictions"
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

            plt.plot(
                x, mean_rmse, label=f"{label}: $\\mu={np.mean(mean_rmse):.3f}$"
            )
            plt.fill_between(x, mean_rmse - sqrt_rmse, mean_rmse + sqrt_rmse, alpha=0.2)

            plt.title(title + ": " + station_dict[station])
            plt.xlabel("$\\text{t}^{th}$ prediction")
            plt.ylabel("RMSE")
            plt.legend()

            if save_dir:
                plt.savefig(os.path.join(f"figures/{save_dir}/stations/{station_dict[station]}.pdf"))

    if not save_dir:
        plt.show()


def plot_scatter(
    root: str, summer: bool, files: List[str], labels: List[str], save_dir: str
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

            plt.figure(figure_count, clear=True)
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
                    marker="x", s=16
                )

                # plt.plot(new_x, y_hat,
                #          color=color_dict[station],
                #          linestyle="--", alpha=0.6)

                plt.title(label + "-" + str(lag) + "-" + title)
                plt.xlabel("Prediction")
                plt.ylabel("Target")
                plt.legend()

            if save_dir:
                os.makedirs(os.path.join(f"figures/{save_dir}/scatter/{label}"), exist_ok=True)
                plt.savefig(os.path.join(f"figures/{save_dir}/scatter/{label}/{lag}.pdf"))

    if not save_dir:
        plt.show()
    else:
        plt.close()


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
    parser.add_argument("--save_dir", type=str, help="Where to store the output")

    args = parser.parse_args()

    main(
        "experiments/", args.summer, args.files, args.labels,
        args.plot_all, args.plot_stations, args.plot_scatter, args.save_dir
    )
