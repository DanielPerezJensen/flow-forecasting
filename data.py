import numpy as np
import pandas as pd
import os
import torch
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer
from typing import Type, Optional, Tuple, List


class GraphRiverFlowDataset():
    def __init__(
                self,
                root: str,
                scaler_name: str = "none",
                freq: str = "M",
                lag: int = 6,
                lagged_variables: Optional[List[str]] = None,
                target_variable: str = "river_flow",
                target_stations: Optional[List[int]] = None
            ):

        self.root = root
        self.freq = freq
        self.scaler_name = scaler_name
        self.lag = lag

        if lagged_variables is None:
            self.lagged_variables = ["river_flow"]
        else:
            self.lagged_variables = lagged_variables

        self.target_variable = target_variable

        if target_stations is None:
            self.target_stations = [0, 1, 2, 3]
        else:
            self.target_stations = target_stations

        self.process()

    def process(self):

        df = load_and_aggregate_flow_data(self.freq)
        df_lagged = generate_lags(df, self.lagged_variables, self.lag)

        # Extract nodes and eddges from disk
        measurements_feats = load_nodes_csv(os.path.join(self.root, "measurement.csv"), self.scaler_name)
        subsubwatersheds_feats = load_nodes_csv(os.path.join(self.root, "subsub.csv"), self.scaler_name)

        msr_flows_msr, msr_flows_msr_attr = load_edges_csv(os.path.join(self.root, "measurement-flows-measurement.csv"), self.scaler_name)
        sub_flows_sub, sub_flows_sub_attr = load_edges_csv(os.path.join(self.root, "subsub-flows-subsub.csv"), self.scaler_name)
        sub_in_msr, _ = load_edges_csv(os.path.join(self.root, "subsub-in-measurement.csv"), self.scaler_name)

        dataset = []

        unique_dates = df_lagged.date.unique()

        for date in unique_dates:
            date_df = df_lagged.loc[df_lagged.date == date].sort_values("station_number")

            # Extract date features and add the static features
            date_features = torch.from_numpy(date_df.loc[:, date_df.columns.str.match(".*_\\d")].to_numpy())
            date_features = torch.cat([measurements_feats, date_features], dim=-1)

            # Extract date targets and convert to tensor
            date_targets_df = date_df.loc[date_df["station_number"].isin(self.target_stations)]
            date_targets = torch.from_numpy(date_targets_df[self.target_variable].to_numpy())

            data = HeteroData()

            data["measurement"].x = date_features
            data["measurement"].y = date_targets
            data["subsub"].x = subsubwatersheds_feats

            data["measurement", "flows", "measurement"].edge_index = msr_flows_msr
            data["measurement", "flows", "measurement"].edge_label = msr_flows_msr_attr

            data["subsub", "flows", "subsub"].edge_index = sub_flows_sub
            data["subsub", "flows", "subsub"].edge_label = sub_flows_sub_attr

            data["subsub", "in", "measurement"].edge_index = sub_in_msr

            dataset.append(data)

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def generate_lags(df: pd.DataFrame, values: list, n_lags: int) -> pd.DataFrame:
    """
    function: generate_lags

    Generates a dataframe with columns denoting lagged value up to n_lags,
    does this per station number so there is no overlap between stations.
    """
    station_numbers = df.station_number.unique()
    frames = []

    # Iterate over dataframes split by station number
    for _, df_station_flow_agg in df.groupby("station_number"):

        # Lag each dataframe individually
        df_n = df_station_flow_agg.copy()

        for value in values:
            for n in range(1, n_lags + 1):
                df_n[f"{value}_{n}"] = df_n[f"{value}"].shift(n)

        frames.append(df_n)

    # Impute missing values
    df_merged_flow_agg = pd.concat(frames)
    df_merged_flow_agg = df_merged_flow_agg.fillna(-1)

    return df_merged_flow_agg


def get_scaler(scaler: str) -> Optional[Type]:
    """
    function: get_scaler

    Returns a scaler from a selection of 4 options, given the string name.
    """
    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "maxabs": MaxAbsScaler(),
        "robust": RobustScaler(),
        "none": FunctionTransformer(pd.DataFrame.to_numpy)
    }

    return scalers[scaler.lower()]


def load_nodes_csv(path, scaler_name="none", **kwargs):
    """
    function: load_nodes_csv

    Loads nodes from provided csv and scales using scaler name.
    Return scaled data and scaler.
    """
    df = pd.read_csv(path, index_col=0, **kwargs)

    scaler = get_scaler(scaler_name)
    scaler.fit(df)

    x = scaler.transform(df)
    x = torch.from_numpy(x)

    return x


def load_edges_csv(path, scaler_name="none", **kwargs):
    df = pd.read_csv(path, index_col=0, **kwargs)

    # Gather edge index as 2 dimensional tensor
    edge_index = torch.from_numpy(df[["src", "dst"]].values.T)
    df_attr = df.drop(["src", "dst"], axis=1)

    # Only scale the edge attributes if there are features in the dataframe
    if len(df_attr.columns) == 1:
        scaler = get_scaler(scaler_name)
        scaler.fit(df_attr)

        edge_attr = scaler.transform(df_attr)
        edge_attr = torch.from_numpy(edge_attr)

    else:
        edge_attr = []

    return edge_index, edge_attr


def load_and_aggregate_flow_data(freq: str = "M") -> pd.DataFrame:
    """
    function: load_and_aggregate_flow_data

    Reads river flow data and aggregates it to the frequency specified in freq.
    Return aggregated river flow data.
    """
    processed_folder_path = os.path.join("data", "processed")

    # Import river flow data and only preserve datapoints after 1965
    df_flow = pd.read_csv(os.path.join(processed_folder_path,
                                       "raw-measurements.csv"),
                          index_col=0, parse_dates=["date"])
    df_flow = df_flow.loc[df_flow["date"].dt.year >= 1965]

    # Gather every month from start and end date
    start_date = df_flow["date"].min()
    end_date = df_flow["date"].max()
    date_range = pd.date_range(start_date, end_date, freq=freq, normalize=True)

    # We split the dataframes based on station number and process them
    station_dfs = []

    for station_number in df_flow.station_number.unique():
        df_station_flow = df_flow.loc[df_flow.station_number == station_number]
        df_station_flow_aggregated = df_station_flow.groupby(
                                pd.Grouper(key="date", freq=freq)
                            )[["river_flow", "river_height"]].mean()
        df_station_flow_aggregated = df_station_flow_aggregated.reset_index()

        # Create new dataframe based on date range so every date is found in
        # flow data
        new_df = pd.DataFrame(date_range, columns=["date"])
        df_station_flow_aggregated = pd.merge(new_df,
                                              df_station_flow_aggregated,
                                              how="left")
        df_station_flow_aggregated["station_number"] = station_number

        station_dfs.append(df_station_flow_aggregated)

    df_flow_aggregated = pd.concat(station_dfs)

    # Fill in missing values with -1 for imputation
    df_flow_aggregated = df_flow_aggregated.fillna(-1)

    return df_flow_aggregated
