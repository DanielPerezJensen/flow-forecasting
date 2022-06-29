from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import os
from datetime import timedelta
from collections import OrderedDict
import re

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer

from omegaconf import DictConfig, OmegaConf

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
        lagged_vars: Optional[List[str]] = None,
        lagged_stations: Optional[List[int]] = None,
        target_var: str = "river_flow",
        target_stations: Optional[List[int]] = None,
        time_features: bool = False,
        ndsi: Optional[DictConfig] = None,
        ndvi: Optional[DictConfig] = None,
        process: bool = False
    ) -> None:
        """
        RiverFlowDataset:

        This class contains all data handling code, aListnd creates a properly
        formatted PyTorch Dataset
        """
        self.root = root if root else os.path.join("data", "processed")

        self.freq = freq
        self.scaler_name = scaler_name

        # Set lag and amount of prediction according to frequency
        if self.freq == "M":
            self.lag = 6
        elif self.freq == "W":
            self.lag = 24

        self.n_preds = self.lag

        self.lagged_vars = lagged_vars if lagged_vars else ["river_flow"]

        self.lagged_stations = lagged_stations if lagged_stations else [34]

        self.target_var = target_var

        if target_stations is None:
            self.target_stations = [34]
        else:
            self.target_stations = target_stations

        self.time_features = time_features

        if ndsi is not None:
            # If any of the features are used store that in boolean
            self.ndsi_features = ndsi.index or ndsi.surface or ndsi.cloud
            self.ndsi_index = ndsi.index
            self.ndsi_surface = ndsi.surface
            self.ndsi_cloud = ndsi.cloud

        if ndvi is not None:
            self.ndvi_features = ndvi.index or ndvi.surface or ndvi.cloud
            self.ndvi_index = ndvi.index
            self.ndvi_surface = ndvi.surface
            self.ndvi_cloud = ndvi.cloud

        self.data_date_dict = OrderedDict()  # type: DataDateDictType

        if process:
            assert self.root
            self.process(self.root)

    def process(self, root: Union[str, os.PathLike[Any]]) -> None:
        """
        This function processes the data found in root into an ordered
        dict and list of data
        Args:
            root: path to preprocessed data
        """
        if not self.root:
            raise ValueError

        # Gather lagged flow data
        df_features = load_and_aggregate_flow_data(self.root, self.freq)

        # Add time features if needed
        if self.time_features:
            df_features, self.lagged_vars = set_time_features(
                self.freq, df_features, self.lagged_vars
            )

        if self.ndsi_features:

            df_features, self.lagged_vars = load_and_aggregate_ndsi_ndvi_data(
                "NDSI", self.root, df_features, self.freq, self.lagged_vars,
                self.ndsi_index, self.ndsi_surface, self.ndsi_cloud
            )

        if self.ndvi_features:

            df_features, self.lagged_vars = load_and_aggregate_ndsi_ndvi_data(
                "NDVI", self.root, df_features, self.freq, self.lagged_vars,
                self.ndvi_index, self.ndvi_surface, self.ndvi_cloud
            )

        df_flow_lagged = generate_lags(df_features, self.lagged_vars, self.lag)
        df_flow_lagged = generate_lags(df_flow_lagged,
                                       [self.target_var], -self.n_preds)

        # Drop any date for which any of the target stations has no measurement
        df_target_stations = df_flow_lagged[
            df_flow_lagged.station_number.isin(self.target_stations)
        ]
        dropped_dates = df_target_stations[
            df_target_stations[self.target_var].isna()
        ]["date"]

        df_flow_lagged = df_flow_lagged.drop(
            df_flow_lagged.index[df_flow_lagged.date.isin(dropped_dates)]
        )

        # Data imputation
        df_flow_lagged = df_flow_lagged.fillna(-1)

        # Scale data
        self.scaler = get_scaler(self.scaler_name)  # type: ScalerType

        flow_scaled_cols = [
            col for col in df_flow_lagged if col not in [
                self.target_var, "date", "station_number"
            ]
        ]

        df_flow_lagged[flow_scaled_cols] = self.scaler.fit_transform(
            df_flow_lagged[flow_scaled_cols]
        )

        # Scale target column separately as we need to inverse transform later
        df_flow_lagged[[self.target_var]] = self.scaler.fit_transform(
            df_flow_lagged[[self.target_var]]
        )

        unique_dates = df_flow_lagged.date.unique()
        df_stations_date = df_features.loc[
            df_features["station_number"].isin(self.lagged_stations)
        ]

        for date in unique_dates:
            date = np.datetime64(date, "D")

            df_flow_date = df_flow_lagged.loc[
                df_flow_lagged.date == date
            ].sort_values("station_number")

            df_stations_date = df_flow_date.loc[
                df_flow_date["station_number"].isin(self.lagged_stations)
            ]

            # Concatenate all lagged variables we want into one input vector
            df_date_features = df_stations_date.loc[
                :, df_stations_date.columns.str.fullmatch("river_flow-\\d+")
            ]

            date_features = torch.from_numpy(
                df_date_features.to_numpy(dtype=np.float32).T
            )

            if self.time_features:
                df_time_features = df_stations_date.loc[
                    :, df_stations_date.columns.str.fullmatch(
                        "((sin)|(cos))_.*\\d+"
                    )
                ]
                # We sort the time features so we get
                # [sin_1, cos_1, sin_2, cos_2, ..., sin_24, cos_24]
                # regex there for more than one digit
                df_time_features = df_time_features[
                    sorted(df_time_features.columns,
                           key=lambda x: int(re.search(r'\d+$', x).group()))
                ]

                # Only add one time feature as they are equal across stations
                time_features = torch.from_numpy(
                    df_time_features.to_numpy(dtype="float32")[0, :][None, :]
                )

                time_features = time_features.reshape((self.lag, -1))
                date_features = torch.cat([date_features, time_features],
                                          dim=1)

            if self.ndsi_features:

                date_features = append_ndsi_ndvi_features(
                    "NDSI", df_stations_date, date_features, self.ndsi_index,
                    self.ndsi_surface, self.ndsi_cloud)

            if self.ndvi_features:

                date_features = append_ndsi_ndvi_features(
                    "NDVI", df_stations_date, date_features, self.ndvi_index,
                    self.ndvi_surface, self.ndvi_cloud)

            # Extract targets
            df_target_date = df_flow_date.loc[
                df_flow_date["station_number"].isin(self.target_stations)
            ]

            df_targets_date = df_target_date.loc[
                :, df_target_date.columns.str.fullmatch("river_flow\\+\\d+")
            ]

            date_targets = torch.from_numpy(
                df_targets_date.to_numpy(dtype="float32")
            )

            # We always want 6 predictions, so aggregate weekly into monthly
            if self.freq == "W":
                date_targets = date_targets.reshape(
                    (len(self.target_stations), -1, 4)
                ).mean(dim=2)

            self.data_date_dict[date] = (date_features.float(),
                                         date_targets.float())

        self.data_list = list(self.data_date_dict.values())

    def set_data(self, date: np.datetime64, value: BatchType) -> None:
        """
        This function will set a datapoint in the OrderedDict by date
        Args:
            date: np.datetime64
            value: batch of data
        """
        self.data_date_dict[date] = value
        self.data_list = list(self.data_date_dict.values())

    def get_item_by_date(self, date: Union[np.datetime64, str]) -> BatchType:
        """
        Returns a batch item using a date from the OrderedDict
        Args:
            date: np.datetime64
        """
        date = np.datetime64(date)
        return self.data_date_dict[date]

    def __repr__(self) -> str:
        return repr(self.data_list)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> BatchType:
        return self.data_list[index]


def split_dataset(
    dataset: RiverFlowDataset, cfg: DictConfig,
    val_year_min: int = 1998, val_year_max: int = 2002,
    test_year_min: int = 2013, test_year_max: int = 2019
) -> Tuple[RiverFlowDataset, RiverFlowDataset, RiverFlowDataset]:

    assert cfg.freq in ["W", "M"]
    assert val_year_min < val_year_max
    assert test_year_min < test_year_max

    if cfg.freq == "M":
        offset = pd.tseries.offsets.DateOffset(months=dataset.lag)
    if cfg.freq == "W":
        offset = pd.tseries.offsets.DateOffset(weeks=dataset.lag)

    train_dataset = RiverFlowDataset(**cfg)
    val_dataset = RiverFlowDataset(**cfg)
    test_dataset = RiverFlowDataset(**cfg)

    val_start = np.datetime64(str(val_year_min), "D")
    val_end = np.datetime64(str(val_year_max), "D")

    test_start = np.datetime64(str(test_year_min), "D")
    test_end = np.datetime64(str(test_year_max), "D")

    for date in dataset.data_date_dict:
        if val_start <= date < val_end - offset:
            val_dataset.set_data(date, dataset.get_item_by_date(date))
        elif test_start <= date < test_end - offset:
            test_dataset.set_data(date, dataset.get_item_by_date(date))
        # These dates are not allowed in the training set as
        # they are found as features in the validation or test set
        elif val_start - offset <= date < val_end:
            continue
        elif test_start - offset <= date < test_end:
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
        df_station_flow_aggregated = pd.merge_ordered(
                                        new_df,
                                        df_station_flow_aggregated,
                                        how="left")
        df_station_flow_aggregated["station_number"] = station_number

        station_dfs.append(df_station_flow_aggregated)

    df_flow_aggregated = pd.concat(station_dfs)

    return df_flow_aggregated


def load_and_aggregate_ndsi_ndvi_data(
    name: str, root: Union[str, os.PathLike[Any]], df_features: pd.DataFrame,
    freq: str, lagged_vars: List[str], index: bool, surface: bool, cloud: bool
) -> Tuple[pd.DataFrame, List[str]]:

    df = gather_ndsi_ndvi_data(root, f"{name}.csv")

    if index:
        index_df = aggregate_index_data(name, freq, df)
        df_features = pd.merge(df_features, index_df, how="left")
        lagged_vars += [f"{name}_avg"]

    if surface:
        surface_df = aggregate_area_data(name, freq, df, "Surfavg")
        df_features = pd.merge(df_features, surface_df, how="left")
        lagged_vars += [f"{name}_Surfavg"]

    if cloud:
        cloud_df = aggregate_area_data(name, freq, df, "Surfcloudavg")
        df_features = pd.merge(df_features, cloud_df, how="left")
        lagged_vars += [f"{name}_Surfcloudavg"]

    return df_features, lagged_vars


def set_time_features(
    freq: str, df_features: pd.DataFrame, lagged_vars: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    if freq == "W":
        df_features = (
            df_features.assign(
                month=df_features.date.dt.month,
                week=df_features.date.dt.isocalendar().week)
        )
        df_features = generate_cyclical_features(df_features,
                                                 "month", 12, 1)
        df_features = generate_cyclical_features(df_features,
                                                 "week", 52, 1)
        lagged_vars += ["sin_month", "cos_month"]
        lagged_vars += ["sin_week", "cos_week"]

    elif freq == "M":
        df_features = (
            df_features.assign(
                month=df_features.date.dt.month)
        )
        df_features = generate_cyclical_features(df_features,
                                                 "month", 12, 1)
        lagged_vars += ["sin_month", "cos_month"]

    return df_features, lagged_vars


def append_ndsi_ndvi_features(
    name: str, df_stations_date: pd.DataFrame,
    date_features: torch.Tensor,
    index: bool, surface: bool, cloud: bool
) -> torch.Tensor:

    if index:
        df_feature = df_stations_date.loc[
            :, df_stations_date.columns.str.fullmatch(f"{name}_avg-\\d+")
        ]

        # We only want one of them as they are copies of each other
        # as we are aggregating over the whole watershed
        feature = torch.from_numpy(
            df_feature.to_numpy(dtype="float32").T[:, 0][:, None]
        )

        date_features = torch.cat((date_features, feature), dim=1)

    if surface:
        df_feature = df_stations_date.loc[
            :, df_stations_date.columns.str.fullmatch(f"{name}_Surfavg-\\d+")
        ]

        # We only want one of them as they are copies of each other
        # as we are aggregating over the whole watershed
        feature = torch.from_numpy(
            df_feature.to_numpy(dtype="float32").T[:, 0][:, None]
        )

        date_features = torch.cat((date_features, feature), dim=1)

    if cloud:
        df_feature = df_stations_date.loc[
            :, df_stations_date.columns.str.fullmatch(
                    f"{name}_Surfcloudavg-\\d+"
                )
        ]

        # We only want one of them as they are copies of each other
        # as we are aggregating over the whole watershed
        feature = torch.from_numpy(
            df_feature.to_numpy(dtype="float32").T[:, 0][:, None]
        )

        date_features = torch.cat((date_features, feature), dim=1)

    return date_features


def generate_lags(
    df: pd.DataFrame, values: List[str], n_lags: int
) -> pd.DataFrame:
    """
    function: generate_lags

    Generates a dataframe with columns denoting lagged value up to n_lags,
    does this per station number so there is no overlap between stations.
    """
    frames = []

    # We use - if the lags are negative and + if the lags are positive
    sig = n_lags / abs(n_lags)
    if sig == 1:
        sign = "-"
    elif sig == -1:
        sign = "+"

    # Iterate over dataframes split by station number
    for _, df_station_flow_agg in df.groupby("station_number"):

        # Lag each dataframe individually
        df_n = df_station_flow_agg.copy()

        # Store added columns and concat after for speediness
        add_columns = []

        for value in values:
            if sign == "-":
                for n in range(1, n_lags + 1):
                    add_columns.append(
                        pd.Series(df_n[f"{value}"].shift(n),
                                  name=f"{value}{sign}{n}")
                    )
            elif sign == "+":
                for n in range(0, -(n_lags)):
                    add_columns.append(
                        pd.Series(df_n[f"{value}"].shift(-n),
                                  name=f"{value}{sign}{n}")
                    )

        add_df = pd.concat(add_columns, axis=1)
        df_n = pd.concat((df_n, add_df), axis=1)

        frames.append(df_n)

    # Impute missing values
    df_merged_flow_agg = pd.concat(frames)

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
    root: Union[str, os.PathLike[Any]],
    filename: str,
    watersheds: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    This function returns the full processed data using various arguments
    as a pd.DataFrame
    Args:
        watersheds: list of strings denoting what watersheds to use from data
        lag: amount of time lag to be used as features
    """
    if watersheds is None:

        watersheds = ["03400", "03401", "03402",
                      "03403", "03404", "03410",
                      "03411", "03412", "03413",
                      "03414", "03420", "03421"]

    df = pd.read_csv(os.path.join(root, filename),
                     index_col=0, parse_dates=["date"],
                     dtype={"Subsubwatershed": str})

    # Only preserve rows inside subsubwatershed list
    keep_rows = df[df.Subsubwatershed.isin(watersheds)].index

    df = df[df.index.isin(keep_rows)]

    return df


def aggregate_area_data(
    name: str, freq: str, df: pd.DataFrame, column: str
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

    grouped_df = df.groupby("date")[[column]].sum()

    # NDSI collected bimonthly so is treated differently
    if name == "NDVI" and freq == "W":
        freq_aggr = grouped_df.resample("D").ffill()[[column]]
        freq_aggr = freq_aggr.groupby(pd.Grouper(freq=freq))[[column]].mean()
    else:
        freq_aggr = grouped_df.groupby(
            pd.Grouper(key='date', freq=freq)
        ).mean()

    freq_aggr = freq_aggr.reset_index()
    freq_aggr = freq_aggr.rename({column: f"{name}_{column}"},
                                 axis="columns")

    return freq_aggr


def aggregate_index_data(
    name: str, freq: str, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns the aggregated NDSI NDVI data with lagged variables
    Args:
        df_NDSI: dataframe containing filtered NDSI values
        df_NDVI: dataframe containing filtered NDVI values
    """
    df = df.sort_values(["date", "Subsubwatershed"])

    # NDSI collected bimonthly so is treated differently
    if name == "NDVI" and freq == "W":
        freq_aggr = df.groupby("date").mean()
        freq_aggr = freq_aggr.resample("D").ffill()[["avg"]]
        freq_aggr = freq_aggr.groupby(pd.Grouper(freq=freq))[["avg"]].mean()
    else:
        # Take average of NDSI values for each month and aggregate
        freq_aggr = df.groupby(
            pd.Grouper(key='date', freq=freq)
        )[["avg"]].mean()

    freq_aggr = freq_aggr.reset_index()
    freq_aggr = freq_aggr.rename({"avg": f"{name}_avg"}, axis="columns")

    return freq_aggr
