import numpy as np
import pandas as pd
import os
from datetime import timedelta
from collections import OrderedDict

import torch
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer

from typing import Type, Optional, Tuple, List, Union, Dict

DataDateDictType = Dict[np.datetime64, HeteroData]


class GraphFlowDataset():
    def __init__(
        self,
        root: Optional[Union[str, os.PathLike]] = None,
        graph_type: Optional[str] = None,
        scaler_name: str = "none",
        freq: str = "M",
        lag: int = 6,
        lagged_vars: Optional[List[str]] = None,
        target_var: str = "river_flow",
        target_stations: Optional[List[int]] = None,
        process: bool = False
    ) -> None:

        self.root = root if root else os.path.join("data", "processed")
        self.graph_type = graph_type if graph_type else "base"

        self.freq = freq
        self.scaler_name = scaler_name
        self.lag = lag

        self.lagged_vars = lagged_vars if lagged_vars else ["river_flow"]

        self.target_var = target_var

        if target_stations is None:
            self.target_stations = [0, 1, 2, 3]
        else:
            self.target_stations = target_stations

        self.data_date_dict = OrderedDict()  # type: DataDateDictType

        if process:
            assert root
            self.process(root)

    def process(self, root: Union[str, os.PathLike]) -> None:

        if not self.root:
            raise ValueError

        df = load_and_aggregate_flow_data(root, self.freq)
        df_lagged = generate_lags(df, self.lagged_vars, self.lag)

        # Drop any rows where target variable is nan
        df_lagged = df_lagged.drop(df.index[df[self.target_var] == -1])

        # Extract nodes and eddges from disk
        measurements_feats = load_nodes_csv(os.path.join(self.root, "static", "measurement.csv"), self.scaler_name)
        subsubwatersheds_feats = load_nodes_csv(os.path.join(self.root, "static", "subsub.csv"), self.scaler_name)

        msr_flows_msr, msr_flows_msr_attr = load_edges_csv(os.path.join(self.root, "graph", self.graph_type, "measurement-flows-measurement.csv"), self.scaler_name)
        sub_flows_sub, sub_flows_sub_attr = load_edges_csv(os.path.join(self.root, "graph", self.graph_type, "subsub-flows-subsub.csv"), self.scaler_name)
        sub_in_msr, _ = load_edges_csv(os.path.join(self.root, "graph", self.graph_type, "subsub-in-measurement.csv"), self.scaler_name)

        unique_dates = df_lagged.date.unique()

        self.scaler = get_scaler(self.scaler_name)

        # Scale all columns besides target and unscalable columns
        scaled_cols = [col for col in df_lagged if col not in [self.target_var, "date", "station_number"]]
        df_lagged[scaled_cols] = self.scaler.fit_transform(df_lagged[scaled_cols])

        # Scale target column separately as we need to inverse transform later
        df_lagged[[self.target_var]] = self.scaler.fit_transform(df_lagged[[self.target_var]])

        for date in unique_dates:
            date = np.datetime64(date, "D")

            date_df = df_lagged.loc[df_lagged.date == date].sort_values("station_number")

            # Extract date features and add the static features
            date_features = torch.from_numpy(date_df.loc[:, date_df.columns.str.match(".*_\\d")].to_numpy())
            date_features = torch.cat([measurements_feats, date_features], dim=-1)

            # Extract date targets and convert to tensor
            date_targets_df = date_df.loc[date_df["station_number"].isin(self.target_stations)]
            date_targets = torch.from_numpy(date_targets_df[self.target_var].to_numpy())

            data = HeteroData()

            data["measurement"].x = date_features.float()
            data["measurement"].y = date_targets.float()
            data["subsub"].x = subsubwatersheds_feats.float()

            data["measurement", "flows", "measurement"].edge_index = msr_flows_msr
            data["measurement", "flows", "measurement"].edge_attr = msr_flows_msr_attr.float()

            data["subsub", "flows", "subsub"].edge_index = sub_flows_sub
            data["subsub", "flows", "subsub"].edge_attr = sub_flows_sub_attr.float()

            data["subsub", "in", "measurement"].edge_index = sub_in_msr

            self.data_date_dict[date] = data

        self.data_list = list(self.data_date_dict.values())

    def set_data(self, date, value):
        self.data_date_dict[date] = value
        self.data_list = list(self.data_date_dict.values())

    def get_item_by_date(self, date) -> HeteroData:
        return self.data_date_dict[date]

    def __repr__(self) -> str:
        return repr(self.data_list)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index) -> HeteroData:
        return self.data_list[index]


def split_dataset(
    dataset: GraphFlowDataset,
    freq: str = "M", lag: int = 6,
    val_year_min: int = 1998, val_year_max: int = 2002,
    test_year_min: int = 2013, test_year_max: int = 2019
) -> Tuple[GraphFlowDataset, GraphFlowDataset, GraphFlowDataset]:

    assert freq in ["W", "M"]
    assert val_year_min < val_year_max
    assert test_year_min < test_year_max

    if freq == "M":
        offset = pd.tseries.offsets.DateOffset(months=lag)
    if freq == "W":
        offset = pd.tseries.offsets.DateOffset(weeks=lag)

    train_dataset = GraphFlowDataset()
    val_dataset = GraphFlowDataset()
    test_dataset = GraphFlowDataset()

    val_start = np.datetime64(str(val_year_min), "D")
    val_end = np.datetime64(str(val_year_max), "D")

    test_start = np.datetime64(str(test_year_min), "D")
    test_end = np.datetime64(str(test_year_max), "D")

    for date in dataset.data_date_dict:
        if val_start <= date < val_end:
            val_dataset.set_data(date, dataset.get_item_by_date(date))
        elif test_start <= date < test_end:
            test_dataset.set_data(date, dataset.get_item_by_date(date))
        # These dates are not allowed in the training set as
        # they are found as features in the validation or test set
        elif val_start - offset <= date < val_start:
            continue
        elif val_end <= date < val_end + offset:
            continue
        elif test_start - offset <= date < test_start:
            continue
        elif test_end <= date < test_end + offset:
            continue
        else:
            train_dataset.set_data(date, dataset.get_item_by_date(date))

    return train_dataset, val_dataset, test_dataset


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


def get_scaler(scaler: str) -> Type:
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


def load_nodes_csv(
    path: Union[str, os.PathLike],
    scaler_name: str = "none", **kwargs
) -> torch.Tensor:
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


def load_edges_csv(
    path: Union[str, os.PathLike],
    scaler_name: str = "none",
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        edge_attr = torch.empty((0))

    return edge_index, edge_attr


def load_and_aggregate_flow_data(
    root: Union[str, os.PathLike],
    freq: str = "M"
) -> pd.DataFrame:
    """
    function: load_and_aggregate_flow_data

    Reads river flow data and aggregates it to the frequency specified in freq.
    Return aggregated river flow data.
    """

    # Import river flow data and only preserve datapoints after 1965
    df_flow = pd.read_csv(os.path.join(root, "temporal",
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


if __name__ == "__main__":
    root = os.path.join("data", "processed")
    dataset = GraphFlowDataset(root, process=True, scaler_name="minmax", freq="W")
