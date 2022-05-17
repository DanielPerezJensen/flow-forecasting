from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import os
from datetime import timedelta
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer

from typing import Type, Optional, Tuple, List, Union, Dict, Any, Callable

ScalerType = Union[MinMaxScaler, StandardScaler,
                   MaxAbsScaler, RobustScaler, FunctionTransformer]
BatchType = Tuple[torch.Tensor, torch.Tensor]
DataDateDictType = Dict[np.datetime64, BatchType]


class RiverFlowDataset(Dataset[Any]):
    def __init__(
        self,
        root: Optional[Union[str, os.PathLike[Any]]] = None,
        scaler_name: str = "none",
        freq: str = "M",
        lag: int = 6,
        lagged_vars: Optional[List[str]] = None,
        lagged_stations: Optional[List[int]] = None,
        target_var: str = "river_flow",
        target_stations: Optional[List[int]] = None,
        time_features: bool = False,
        process: bool = False
    ) -> None:
        self.root = root if root else os.path.join("data", "processed")

        self.freq = freq
        self.scaler_name = scaler_name
        self.lag = lag

        self.lagged_vars = lagged_vars if lagged_vars else ["river_flow"]

        self.lagged_stations = lagged_stations if lagged_stations else [34]

        self.target_var = target_var

        if target_stations is None:
            self.target_stations = [34]
        else:
            self.target_stations = target_stations

        self.time_features = time_features

        self.data_date_dict = OrderedDict()  # type: DataDateDictType

        if process:
            assert self.root
            self.process(self.root)

    def process(self, root: Union[str, os.PathLike[Any]]) -> None:
        if not self.root:
            raise ValueError

        # Gather lagged flow data
        df_flow_aggregated = load_and_aggregate_flow_data(self.root, self.freq)
        df_flow_lagged = generate_lags(df_flow_aggregated, self.lagged_vars,
                                       self.lag)

        # Drop nan values from target variable (river_flow)
        df_flow_lagged = df_flow_lagged.drop(
            df_flow_lagged.index[df_flow_lagged[self.target_var] == -1]
        )

        # Add time features if needed
        if self.time_features:
            df_flow_lagged = (
                df_flow_lagged.assign(
                    month=df_flow_lagged.date.dt.month,
                    week=df_flow_lagged.date.dt.isocalendar().week)
            )
            # For the weekly scale we can use both month and
            # week as a cyclical feature
            if self.freq == "M" or self.freq == "W":
                df_flow_lagged = generate_cyclical_features(df_flow_lagged,
                                                            "month", 12, 1)
            if self.freq == "W":
                df_flow_lagged = generate_cyclical_features(df_flow_lagged,
                                                            "week", 52, 1)

        # Scale data
        self.scaler = get_scaler(self.scaler_name)  # type: ScalerType

        # Scale all columns besides target and unscalable columns
        flow_scaled_cols = [
            col for col in df_flow_lagged if col not in [self.target_var, "date", "station_number"]
        ]
        df_flow_lagged[flow_scaled_cols] = self.scaler.fit_transform(
            df_flow_lagged[flow_scaled_cols]
        )

        # Scale target column separately as we need to inverse transform later
        df_flow_lagged[[self.target_var]] = self.scaler.fit_transform(
            df_flow_lagged[[self.target_var]]
        )

        unique_dates = df_flow_lagged.date.unique()

        self.time_features = True

        for date in unique_dates:
            date = np.datetime64(date, "D")

            df_flow_date = df_flow_lagged.loc[df_flow_lagged.date == date].sort_values("station_number")

            df_stations_date = df_flow_date.loc[df_flow_date["station_number"].isin(self.lagged_stations)]

            # Concatenate all lagged variables we want into one input vector
            df_date_features = df_stations_date.loc[:, df_stations_date.columns.str.match(".*\\d")]
            date_features = df_date_features.to_numpy(dtype=np.float32).flatten()

            if self.time_features:
                df_time_features = df_stations_date.loc[:, df_stations_date.columns.str.match("(sin)|(cos)_.*")]
                # Only add one time feature as they are the same across the stations
                date_features = np.append(date_features, df_time_features.to_numpy(dtype=np.float32)[0, :])

            date_features = torch.from_numpy(date_features)

            df_target_date = df_flow_date.loc[df_flow_date["station_number"].isin(self.target_stations)]
            date_targets = torch.from_numpy(df_target_date[self.target_var].to_numpy(dtype=np.float32))

            self.data_date_dict[date] = (date_features, date_targets)

        self.data_list = list(self.data_date_dict.values())

    def set_data(self, date: np.datetime64, value: BatchType) -> None:
        self.data_date_dict[date] = value
        self.data_list = list(self.data_date_dict.values())

    def get_item_by_date(self, date: np.datetime64) -> BatchType:
        return self.data_date_dict[date]

    def __repr__(self) -> str:
        return repr(self.data_list)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> BatchType:
        return self.data_list[index]


def split_dataset(
    dataset: RiverFlowDataset,
    freq: str = "M", lag: int = 6,
    val_year_min: int = 1998, val_year_max: int = 2002,
    test_year_min: int = 2013, test_year_max: int = 2019
) -> Tuple[RiverFlowDataset, RiverFlowDataset, RiverFlowDataset]:

    assert freq in ["W", "M"]
    assert val_year_min < val_year_max
    assert test_year_min < test_year_max

    if freq == "M":
        offset = pd.tseries.offsets.DateOffset(months=lag)
    if freq == "W":
        offset = pd.tseries.offsets.DateOffset(weeks=lag)

    train_dataset = RiverFlowDataset()
    val_dataset = RiverFlowDataset()
    test_dataset = RiverFlowDataset()

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


def load_and_aggregate_flow_data(
    root: Union[str, os.PathLike[Any]],
    freq: str = "M"
) -> pd.DataFrame:
    """
    function: load_and_aggregate_flow_data

    Reads river flow data and aggregates it to the frequency specified in freq.
    Return aggregated river flow data.
    """

    # Import river flow data and only preserve datapoints after 1965
    df_flow = pd.read_csv(os.path.join(root, "raw-measurements.csv"),
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


def generate_lags(
    df: pd.DataFrame, values: List[str], n_lags: int
) -> pd.DataFrame:
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


def generate_cyclical_features(
    df: pd.DataFrame, col_name: str, period: int, start_num: int = 0
) -> pd.DataFrame:

    kwargs = {
        f"sin_{col_name}":
            lambda x: np.sin(2 * np.pi * (df[col_name] - start_num) / period),
        f"cos_{col_name}":
            lambda x: np.cos(2 * np.pi * (df[col_name] - start_num) / period)
    }

    return df.assign(**kwargs).drop(columns=[col_name])


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


def gather_ndsi_ndvi_data(
    watersheds: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function returns the full processed data using various arguments
    as a pd.DataFrame
    Args:
        watersheds: list of strings denoting what watersheds to use from data
        lag: amount of time lag to be used as features
    """
    processed_folder_path = os.path.join("data", "processed")

    if watersheds is None:

        watersheds = ["03400", "03401", "03402",
                      "03403", "03404", "03410",
                      "03411", "03412", "03413",
                      "03414", "03420", "03421"]

    df_NDSI = pd.read_csv(os.path.join(processed_folder_path, "NDSI.csv"),
                          index_col=0, parse_dates=["date"],
                          dtype={"Subsubwatershed": str})
    df_NDVI = pd.read_csv(os.path.join(processed_folder_path, "NDVI.csv"),
                          index_col=0, parse_dates=["date"],
                          dtype={"Subsubwatershed": str})

    # Only preserve rows inside subsubwatershed list
    keep_rows_ndsi = df_NDSI[df_NDSI.Subsubwatershed.isin(watersheds)].index
    keep_rows_ndvi = df_NDVI[df_NDVI.Subsubwatershed.isin(watersheds)].index

    df_NDSI = df_NDSI[df_NDSI.index.isin(keep_rows_ndsi)]
    df_NDVI = df_NDVI[df_NDVI.index.isin(keep_rows_ndvi)]

    return df_NDSI, df_NDVI


def aggregate_area_data(
    freq: str, df_NDSI: pd.DataFrame, df_NDVI: pd.DataFrame, column: str
) -> pd.DataFrame:
    """
    This function will correctly aggregate area data given the column
    Args:
        freq: frequency of aggregation
        df_NDSI: dataframe containing filtered NDSI values
        df_NDVI: dataframe containing filtered NDVI values
        column: column name to aggregate, must contain 'Surf'
    """
    assert "Surf" in column

    # Take sum of each day and average over the months to aggregate area data
    daily_ndsi_surf_sum = df_NDSI.groupby(
                            pd.Grouper(key='date', freq="D")
                        )[[column]].sum().reset_index()
    daily_ndvi_surf_sum = df_NDVI.groupby(
                            pd.Grouper(key='date', freq="D")
                        )[[column]].sum().reset_index()

    monthly_ndsi_surf_mean = daily_ndsi_surf_sum.groupby(
                                pd.Grouper(key='date', freq=freq)
                            )[[column]].mean()
    monthly_ndvi_surf_mean = daily_ndvi_surf_sum.groupby(
                                pd.Grouper(key='date', freq=freq)
                            )[[column]].mean()

    surf_ndsi_mean_df = monthly_ndsi_surf_mean.reset_index()
    surf_ndvi_mean_df = monthly_ndvi_surf_mean.reset_index()
    surf_ndsi_mean_df = surf_ndsi_mean_df.rename({column: f"ndsi_{column}"},
                                                 axis="columns")
    surf_ndvi_mean_df = surf_ndvi_mean_df.rename({column: f"ndvi_{column}"},
                                                 axis="columns")

    surf_ndsi_ndvi_df = pd.merge(surf_ndsi_mean_df, surf_ndvi_mean_df)

    return surf_ndsi_ndvi_df


def aggregate_index_data(
    freq: str, df_NDSI: pd.DataFrame, df_NDVI: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns the aggregated NDSI NDVI data with lagged variables
    Args:
        df_NDSI: dataframe containing filtered NDSI values
        df_NDVI: dataframe containing filtered NDVI values
    """

    # Take average of NDSI values for each month and aggregate
    monthly_ndsi_mean = df_NDSI.groupby(pd.Grouper(
                            key='date', freq=freq)
                        )[["avg"]].mean()
    monthly_ndvi_mean = df_NDVI.groupby(pd.Grouper(
                            key='date', freq=freq)
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
    x = RiverFlowDataset(freq="M", lag=6, lagged_stations=[34, 340, 341, 342], target_stations=[34], scaler_name="standard", process=True, time_features=True)
    sample = x[0]

    print(sample[0].shape)
    print(sample[1].shape)