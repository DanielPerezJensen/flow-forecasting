from __future__ import annotations

import numpy as np
import pandas as pd
import os
from os.path import join
from datetime import timedelta
from collections import OrderedDict


import torch
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer

from typing import Type, Optional, Tuple, List, Union, Dict, Any, Callable

DataDateDictType = Dict[np.datetime64, HeteroData]
ScalerType = Union[MinMaxScaler, StandardScaler,
                   MaxAbsScaler, RobustScaler, FunctionTransformer]


class GraphFlowDataset(Dataset):
    def __init__(
        self,
        root: Optional[Union[str, os.PathLike[Any]]] = None,
        graph_type: Optional[str] = None,
        scaler_name: str = "none",
        freq: str = "M",
        lag: int = 6,
        lagged_vars: Optional[List[str]] = None,
        target_var: str = "river_flow",
        target_stations: Optional[List[int]] = None,
        index_features: bool = False,
        surface_features: bool = False,
        cloud_features: bool = False,
        process: bool = False
    ) -> None:
        self.root = root if root else join("data", "processed")
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

        self.index_features = index_features
        self.surface_features = surface_features
        self.cloud_features = cloud_features

        if process:
            assert root
            self.process(root)

    def process(self, root: Union[str, os.PathLike[Any]]) -> None:
        """
        function: process

        Processes the data found in root
        """
        if not self.root:
            raise ValueError

        # Gather and lag flow data
        df_features = load_and_aggregate_flow_data(root, self.freq)
        df_lagged = generate_lags(df_features, self.lagged_vars, self.lag, groupby_col="station_number")

        # Gather and lag NDSI NDVI values if called for
        if self.index_features or self.surface_features or self.cloud_features:
            df_NDSI_NDVI = gather_ndsi_ndvi_data(
                                self.freq, self.index_features,
                                self.surface_features, self.cloud_features
                            )

            lagged_ndsi_ndvi_vars = df_NDSI_NDVI.columns[df_NDSI_NDVI.columns.str.match(".*(ndsi)|(ndvi).*")]
            df_ndsi_ndvi_lagged = generate_lags(df_NDSI_NDVI, lagged_ndsi_ndvi_vars, self.lag, groupby_col="Subsubwatershed")

            n_subsubs = len(df_ndsi_ndvi_lagged.Subsubwatershed.unique())
            n_lagged_variables = df_ndsi_ndvi_lagged.loc[:, df_ndsi_ndvi_lagged.columns.str.match(".*_\\d")].shape[1]

        # Extract nodes and eddges from disk
        measurements_feats = load_nodes_csv(join(self.root, "static", "measurement.csv"), self.scaler_name)
        subsubwatersheds_feats = load_nodes_csv(join(self.root, "static", "subsub.csv"), self.scaler_name)

        msr_flows_msr, msr_flows_msr_attr = load_edges_csv(join(self.root, "graph", self.graph_type, "measurement-flows-measurement.csv"), self.scaler_name)
        sub_flows_sub, sub_flows_sub_attr = load_edges_csv(join(self.root, "graph", self.graph_type, "subsub-flows-subsub.csv"), self.scaler_name)
        sub_in_msr, _ = load_edges_csv(join(self.root, "graph", self.graph_type, "subsub-in-measurement.csv"), self.scaler_name)

        self.scaler = get_scaler(self.scaler_name)

        # Scale all columns besides target and unscalable columns
        scaled_cols = [col for col in df_lagged if col not in [self.target_var, "date", "station_number"]]
        df_lagged[scaled_cols] = self.scaler.fit_transform(df_lagged[scaled_cols])

        # Scale target column separately as we need to inverse transform later
        df_lagged[[self.target_var]] = self.scaler.fit_transform(df_lagged[[self.target_var]])

        df_lagged = df_lagged.drop(df_lagged.index[df_lagged[self.target_var] == -1])
        unique_dates = df_lagged.date.unique()

        for date in unique_dates:

            date = np.datetime64(date, "D")
            df_date = df_lagged.loc[df_lagged.date == date].sort_values("station_number")
            # Extract date features and add the static features
            date_features = torch.from_numpy(df_date.loc[:, df_date.columns.str.match(f"(({')|('.join(self.lagged_vars)}))_\\d")].to_numpy())
            date_features = torch.cat([measurements_feats, date_features], dim=-1)

            # Extract ndsi/ndvi values
            if self.index_features or self.surface_features or self.cloud_features:
                df_ndsi_ndvi_date = df_ndsi_ndvi_lagged.loc[df_ndsi_ndvi_lagged.date == date]

                # In case we have no NDSI NDVI data we add imputed values
                if len(df_ndsi_ndvi_date) == 0:
                    date_ndsi_ndvi_features = -torch.ones((n_subsubs, n_lagged_variables))
                else:
                    date_ndsi_ndvi_features = torch.from_numpy(df_ndsi_ndvi_date.loc[:, df_ndsi_ndvi_date.columns.str.match(".*_\\d")].to_numpy())

                subsub_features = torch.cat([subsubwatersheds_feats, date_ndsi_ndvi_features], dim=-1)

            else:
                subsub_features = subsubwatersheds_feats

            # Extract date targets and convert to tensor
            df_date_targets = df_date.loc[df_date["station_number"].isin(self.target_stations)]
            date_targets = torch.from_numpy(df_date_targets[self.target_var].to_numpy())

            data = HeteroData()

            data["measurement"].x = date_features.float()
            data["measurement"].y = date_targets.float()
            data["subsub"].x = subsub_features.float()

            data["measurement", "flows", "measurement"].edge_index = msr_flows_msr
            data["measurement", "flows", "measurement"].edge_attr = msr_flows_msr_attr.float()

            data["subsub", "flows", "subsub"].edge_index = sub_flows_sub
            data["subsub", "flows", "subsub"].edge_attr = sub_flows_sub_attr.float()

            data["subsub", "in", "measurement"].edge_index = sub_in_msr

            self.data_date_dict[date] = data

        self.data_list = list(self.data_date_dict.values())

    def set_data(self, date: np.datetime64, value: HeteroData) -> None:
        """
        class function: set_data

        Sets a certain value (or graph) on a certain date in the dataset
        """
        self.data_date_dict[date] = value
        self.data_list = list(self.data_date_dict.values())

    def get_item_by_date(self, date: np.datetime64) -> HeteroData:
        """
        class function: get_item_by_date

        Returns graph based on the date provided
        """
        return self.data_date_dict[date]

    def __repr__(self) -> str:
        return repr(self.data_list)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> HeteroData:
        return self.data_list[index]


def split_dataset(
    dataset: GraphFlowDataset,
    freq: str, lag: int,
    val_year_min: int = 1998, val_year_max: int = 2002,
    test_year_min: int = 2013, test_year_max: int = 2019
) -> Tuple[GraphFlowDataset, GraphFlowDataset, GraphFlowDataset]:
    """
    function: split_dataset

    Splits dataset into training, validation and testing set according to
    the years provided.
    """

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


def generate_lags(
    df: pd.DataFrame, values: List[str],
    n_lags: int, groupby_col: str = "station_number"
) -> pd.DataFrame:
    """
    function: generate_lags

    Generates a dataframe with columns denoting lagged value up to n_lags,
    does this per station number so there is no overlap between stations.
    """
    frames = []

    # Iterate over dataframes split by groupby_col
    for _, group in df.groupby(groupby_col):

        # Lag each dataframe individually
        df_n = group.copy()

        # Store added columns and concat after for speediness
        add_columns = []

        for value in values:
            for n in range(1, n_lags + 1):
                add_columns.append(
                    pd.Series(df_n[f"{value}"].shift(n), name=f"{value}_{n}")
                )

        add_df = pd.concat(add_columns, axis=1)
        df_n = pd.concat((df_n, add_df), axis=1)

        frames.append(df_n)

    # Impute missing values
    df_merged_flow_agg = pd.concat(frames)
    df_merged_flow_agg = df_merged_flow_agg.fillna(-1)

    return df_merged_flow_agg


def get_scaler(scaler: str) -> ScalerType:
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
    path: Union[str, os.PathLike[Any]],
    scaler_name: str = "none", **kwargs: Any
) -> torch.Tensor:
    """
    function: load_nodes_csv

    Loads nodes from provided csv and scales using scaler name.
    """
    df = pd.read_csv(path, index_col=0, **kwargs)

    scaler = get_scaler(scaler_name)
    scaler.fit(df)

    np_x = scaler.transform(df)
    tensor_x = torch.from_numpy(np_x)

    return tensor_x


def load_edges_csv(
    path: Union[str, os.PathLike[Any]],
    scaler_name: str = "none",
    **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    function: load_edges_csv

    Loads edges and edge attributes from provided csv and scales appropiately.
    """
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
    root: Union[str, os.PathLike[Any]],
    freq: str = "M"
) -> pd.DataFrame:
    """
    function: load_and_aggregate_flow_data

    Reads river flow data and aggregates it to the frequency specified in freq.
    """

    # Import river flow data and only preserve datapoints after 1965
    df_flow = pd.read_csv(join(root, "temporal", "raw-measurements.csv"),
                          index_col=0, parse_dates=["date"])
    df_flow = df_flow.loc[df_flow["date"].dt.year >= 1965]

    # Gather every month from start and end date
    date_range = pd.date_range("1965", "2022", freq=freq, normalize=True)

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


def gather_ndsi_ndvi_data(
    freq: str = "W", index_features: bool = True,
    surface_features: bool = False, cloud_features: bool = False
) -> pd.DataFrame:
    """
    function: gather_ndsi_ndvi_data

    Gather ndsi ndvi data and returns whatever is indicated by boolean flags
    """
    processed_folder_path = join("data", "processed")

    df_NDSI = pd.read_csv(
                join(processed_folder_path, "temporal", "raw-NDSI.csv"),
                index_col=0, parse_dates=["date"]
            )
    df_NDVI = pd.read_csv(
                join(processed_folder_path, "temporal", "raw-NDVI.csv"),
                index_col=0, parse_dates=["date"]
            )

    date_range = pd.date_range("1965", "2022", freq=freq)
    df_ndsi_ndvi = pd.DataFrame({"date": date_range}).sort_values("date")

    if index_features:
        df_index_ndsi_ndvi = aggregate_index_data(freq, df_NDSI, df_NDVI)
        df_ndsi_ndvi = pd.merge(df_ndsi_ndvi, df_index_ndsi_ndvi, how="left")

    if surface_features:
        df_surfavg_ndsi_ndvi = aggregate_area_data(freq, df_NDSI,
                                                   df_NDVI, "Surfavg")
        df_ndsi_ndvi = pd.merge(df_ndsi_ndvi, df_surfavg_ndsi_ndvi, how="left")

    if cloud_features:
        df_cloud_ndsi_ndvi = aggregate_area_data(freq, df_NDSI,
                                                 df_NDVI, "Surfcloudavg")
        df_ndsi_ndvi = pd.merge(df_ndsi_ndvi, df_cloud_ndsi_ndvi, how="left")

    df_ndsi_ndvi = df_ndsi_ndvi.dropna()

    return df_ndsi_ndvi


def aggregate_area_data(
    freq: str, df_NDSI: pd.DataFrame, df_NDVI: pd.DataFrame, column: str
) -> pd.DataFrame:
    """
    function: aggregate_area_data

    This function correctly aggregates area data given the column
    """
    assert "Surf" in column

    # Take sum of each day and average over the months to aggregate area data
    daily_ndsi_surf_sum = df_NDSI.groupby(
                            ["Subsubwatershed",
                             pd.Grouper(key='date', freq="D")]
                        )[[column]].sum().reset_index()
    daily_ndvi_surf_sum = df_NDVI.groupby(
                            ["Subsubwatershed",
                             pd.Grouper(key='date', freq="D")]
                        )[[column]].sum().reset_index()

    freq_ndsi_surf_mean = daily_ndsi_surf_sum.groupby(
                            ["Subsubwatershed",
                             pd.Grouper(key='date', freq=freq)]
                            )[[column]].mean()
    freq_ndvi_surf_mean = daily_ndvi_surf_sum.groupby(
                            ["Subsubwatershed",
                             pd.Grouper(key='date', freq=freq)]
                            )[[column]].mean()

    surf_ndsi_mean_df = freq_ndsi_surf_mean.reset_index()
    surf_ndvi_mean_df = freq_ndvi_surf_mean.reset_index()
    surf_ndsi_mean_df = surf_ndsi_mean_df.rename({column: f"ndsi_{column}"},
                                                 axis="columns")
    surf_ndvi_mean_df = surf_ndvi_mean_df.rename({column: f"ndvi_{column}"},
                                                 axis="columns")
    # Merge ndvi and ndsi dataframes into one
    surf_ndsi_ndvi_df = pd.merge(surf_ndsi_mean_df, surf_ndvi_mean_df)

    return surf_ndsi_ndvi_df


def aggregate_index_data(
    freq: str, df_NDSI: pd.DataFrame, df_NDVI: pd.DataFrame
) -> pd.DataFrame:
    """
    function: aggregate_index_data

    Aggregates raw index data
    """

    # Take average of NDSI values for each month and aggregate
    monthly_ndsi_mean = df_NDSI.groupby(
                            ["Subsubwatershed",
                             pd.Grouper(key='date', freq=freq)]
                        )[["avg"]].mean()
    monthly_ndvi_mean = df_NDVI.groupby(
                            ["Subsubwatershed",
                             pd.Grouper(key='date', freq=freq)]
                        )[["avg"]].mean()

    # Rename columns to enable merging
    ndsi_mean_df = monthly_ndsi_mean.reset_index()
    ndvi_mean_df = monthly_ndvi_mean.reset_index()

    ndsi_mean_df = ndsi_mean_df.rename({"avg": "ndsi_avg"}, axis="columns")
    ndvi_mean_df = ndvi_mean_df.rename({"avg": "ndvi_avg"}, axis="columns")

    # Merge ndvi and ndsi dataframes into one
    ndsi_ndvi_df = pd.merge(ndsi_mean_df, ndvi_mean_df)

    return ndsi_ndvi_df


if __name__ == "__main__":
    root = join("data", "processed")
    dataset = GraphFlowDataset(root, process=True, scaler_name="none", freq="W", lag=26, index_features=True)

    x = next(iter(dataset))

    print(x)
