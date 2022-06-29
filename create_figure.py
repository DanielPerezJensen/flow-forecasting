import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from datetime import datetime
import argparse

from typing import Any, List


plt.style.use(['science', 'ieee'])
plt.rcParams.update({'figure.dpi': '100'})


def main(root: str, summer: bool, files: List[str], labels: List[str]) -> None:

    for file, label in zip(files, labels):
        file_dir = os.path.join(root, file)

        if summer:
            title = "RMSE for prediction of summer months"
            filename = "summer-rmse.npy"
            # plt.figure(figsize=(10, 6))
            months = [10, 11, 12, 1, 2, 3]
            x = [datetime(1, int(m), 1).strftime("%b") for m in months]
        else:
            title = "RMSE for prediction of all months in dataset"
            filename = "rmse.npy"
            x = np.arange(6)

        with open(os.path.join(file_dir, filename), "rb") as f:
            rmses = np.load(f)
            mean = np.mean(rmses)

            plt.plot(x, rmses, label=f"{label}: mean={mean:0.3f}")

    plt.title(title)
    plt.xlabel("$\\text{n\\textsuperscript{th}}$ prediction")
    plt.ylabel("RMSE")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--summer", action="store_true")
    parser.add_argument('--files', metavar='N', type=str, nargs='+',
                        help='Filenames')
    parser.add_argument('--labels', metavar='N', type=str, nargs='+',
                        help='Labels')
    parser.add_argument

    args = parser.parse_args()

    main("experiments/", args.summer, args.files, args.labels)
