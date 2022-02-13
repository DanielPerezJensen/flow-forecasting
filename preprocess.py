import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import collections
import os


def main():

    processed_folder_path = os.path.join("data", "processed")
    unprocessed_folder_path = os.path.join("data", "unprocessed")

    # Take care of the NDSI NDVI data
    data_folder = "Data_NDSI_NDVI"
    data_files = ["NDSI.txt", "NDVI.txt"]

    for data_file in data_files:
        df = pd.read_csv(os.path.join(unprocessed_folder_path, data_folder,
                                      data_file),
                         delimiter="\t", index_col=False,
                         names=["Watershed", "Subsubwatershed", "Product",
                                "Date", "Areaini", "Areareproj", f"Surfmax",
                                f"Surfmin", "Surfavg", "max", "min", f"avg",
                                "Surfcloudmax", "Surfcloudmin", "Surfcloudavg",
                                "Surfbadpixmax", "Surfbadpixmin",
                                "Surfbadpixavg"],
                         dtype={"Subsubwatershed": str})

        # Convert date to datetime and convert to wateryears
        df["date"] = pd.to_datetime(df["Date"])
        df = df.drop("Date", axis=1)

        df.to_csv(os.path.join(processed_folder_path, data_file[:4] + ".csv"))

    # Take care of the river flow data
    flow_data_folder = "Data_RiverFlow"
    flow_data_file = "DGA.txt"

    date_columns = ["day", "month", "year", "hour"]

    df = pd.read_csv(os.path.join(unprocessed_folder_path, flow_data_folder,
                                  flow_data_file),
                     delimiter="\t", index_col=False,
                     names=["station_number", "day", "month", "year", "hour",
                            "river_height", "river_flow",
                            "information", "origin"])

    # Convert date go datetime and add as column
    date = pd.to_datetime(dict(year=df.year, month=df.month,
                               day=df.day, hour=df.hour))
    df = df.drop(columns=date_columns)
    df.insert(1, 'date', date)

    df.to_csv(os.path.join(processed_folder_path, flow_data_file[:3] + ".csv"))


if __name__ == "__main__":
    main()
