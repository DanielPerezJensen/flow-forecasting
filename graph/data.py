from __future__ import annotations

import numpy as np
import pandas as pd
import os
from os.path import join
from datetime import timedelta
from collections import OrderedDict
from tqdm import tqdm

import torch
from torch_geometric.data import HeteroData, Dataset
from torch_geometric.loader import DataLoader

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import FunctionTransformer

from omegaconf import DictConfig, OmegaConf

from typing import Type, Optional, Tuple, List, Union, Dict, Any, Callable
from torch_geometric.typing import EdgeType, NodeType, QueryType
from torch_geometric.data.storage import BaseStorage, EdgeStorage, NodeStorage


DataDateDictType = Dict[np.datetime64, HeteroData]
ScalerType = Union[MinMaxScaler, StandardScaler,
                   MaxAbsScaler, RobustScaler, FunctionTransformer]
NodeOrEdgeType = Union[NodeType, EdgeType]
NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

MappingType = Dict[NodeOrEdgeType, Dict[str, torch.Tensor]]


class HeteroSeqData(HeteroData):
    """
    This class defines a special version of HeteroData wherein a
    temporal dimensions is contained within the indices and the xs.
    """
    def __init__(
        self,
        mapping: Optional[MappingType] = None,
        lag: Optional[int] = None, **kwargs: Any
    ):
        super().__init__(mapping, **kwargs)

        # Increment indices for segmentation of subgraphs
        if mapping:
            for key, value in mapping.items():
                # Only consider edges
                if isinstance(key, tuple):
                    src_type, _, dst_type = key
                    edge_indices = value["edge_indices"]
                    new_edge_indices = edge_indices.clone()

                    src_xs = mapping[src_type]["xs"]
                    dst_xs = mapping[dst_type]["xs"]

                    # Increment indices
                    for i, edge_index in enumerate(edge_indices):
                        new_edge_indices[i][0] = (
                            edge_index[0] + (src_xs[0].size(0) * i)
                        )

                        new_edge_indices[i][1] = (
                            edge_index[1] + (dst_xs[0].size(0) * i)
                        )

                    mapping[key]["edge_indices"] = new_edge_indices

            # Initialize HeteroData with properly incremented mapping
            super().__init__(mapping, **kwargs)

        self.lag = lag

    def __inc__(
        self, key: str, value: Any, store: Optional[NodeOrEdgeStorage],
        *args: Any, **kwargs: Any
    ) -> Any:
        if key == "edge_indices" and store:
            src_type, _, dst_type = store._key
            return torch.tensor([[self[src_type].xs[0].size(0)],
                                 [self[dst_type].xs[0].size(0)]]) * self.lag
        else:
            return super().__inc__(key, value, store, *args, **kwargs)

    def __cat_dim__(
        self, key: str, value: Any,
        store: Optional[NodeOrEdgeStorage] = None,
        *args: Any, **kwargs: Any
    ) -> Any:
        if key == "xs":
            return None
        if key == "y":
            return None
        else:
            return super().__cat_dim__(key, value, store, *args, **kwargs)


class GraphFlowDataset(Dataset):
    def __init__(
        self,
        root: Optional[Union[str, os.PathLike[Any]]] = None,
        graph_type: Optional[str] = None,
        scaler_name: str = "none",
        freq: str = "M",
        sequential: bool = False,
        lagged_vars: Optional[List[str]] = None,
        ndsi: Optional[DictConfig] = None,
        ndvi: Optional[DictConfig] = None,
        process: bool = False
    ) -> None:
        self.root = root if root else join("data", "processed")
        self.graph_type = graph_type if graph_type else "base"

        self.freq = freq
        self.scaler_name = scaler_name

        # Set lag and amount of prediction according to frequency
        if self.freq == "M":
            self.lag = 6
        elif self.freq == "W":
            self.lag = 24

        self.n_preds = self.lag

        self.sequential = sequential

        self.lagged_vars = lagged_vars if lagged_vars else ["river_flow"]

        self.target_var = "river_flow"

        self.target_stations = [0, 1, 2, 3]

        self.data_date_dict = OrderedDict()  # type: DataDateDictType

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

        if process:
            assert root
            self.process(root)

    def process(self, root: Union[str, os.PathLike[Any]]) -> None:
        """
        function: process

        Processes the data found in root
        """
        # Gather and lag flow data
        df_features = load_and_aggregate_flow_data(root, self.freq)
        df_lagged = generate_lags(df_features, self.lagged_vars,
                                  self.lag, "station_number")
        df_lagged = generate_lags(df_lagged, [self.target_var],
                                  -self.n_preds, "station_number")

        # Drop any row for which one of the target stations has no measurement
        df_target_stations = df_lagged[
            df_lagged.station_number.isin(self.target_stations)
        ]
        dropped_dates = df_target_stations[
            df_target_stations[self.target_var].isna()
            ]["date"]

        df_lagged = df_lagged.drop(
            df_lagged.index[df_lagged.date.isin(dropped_dates)]
        )

        # Impute nan values
        df_lagged = df_lagged.fillna(-1)

        # Gather ndsi and ndvi features if specified in config
        if self.ndsi_features or self.ndvi_features:
            if self.ndsi_features:
                df_NDSI = load_and_aggregate_ndsi_ndvi_data(
                        root, "NDSI", self.freq, self.ndsi_index,
                        self.ndsi_surface, self.ndsi_cloud
                    )
                NDSI_lagged_cols = df_NDSI.columns[
                    df_NDSI.columns.str.contains("NDSI")
                ]

                df_NDSI_lagged = generate_lags(df_NDSI, NDSI_lagged_cols,
                                               self.lag, "Subsubwatershed")
                df_NDSI_lagged = df_NDSI_lagged.fillna(-1)

            if self.ndvi_features:
                df_NDVI = load_and_aggregate_ndsi_ndvi_data(
                        root, "NDVI", self.freq, self.ndvi_index,
                        self.ndvi_surface, self.ndvi_cloud
                    )

                NDVI_lagged_cols = df_NDVI.columns[
                    df_NDVI.columns.str.contains("NDVI")
                ]

                df_NDVI_lagged = generate_lags(df_NDVI, NDVI_lagged_cols,
                                               self.lag, "Subsubwatershed")
                df_NDVI_lagged = df_NDVI_lagged.fillna(-1)

        # Extract nodes and eddges from disk
        static_msr_feats = load_nodes_csv(
            join(self.root, "static", "measurement.csv"), self.scaler_name
        )
        static_subsub_feats = load_nodes_csv(
            join(self.root, "static", "subsub.csv"), self.scaler_name
        )
        n_static_msr_feats = static_msr_feats.size(1)
        n_static_subsub_feats = static_subsub_feats.size(1)
        n_subsubs = static_subsub_feats.size(0)

        # Edge attributes always use standard scaler
        msr_flows_msr, msr_flows_msr_attr = load_edges_csv(
            join(self.root, "graph", self.graph_type,
                 "measurement-flows-measurement.csv"),
            "standard"
        )

        sub_flows_sub, sub_flows_sub_attr = load_edges_csv(
            join(self.root, "graph", self.graph_type,
                 "subsub-flows-subsub.csv"),
            "standard"
        )

        sub_in_msr, _ = load_edges_csv(
            join(self.root, "graph", self.graph_type,
                 "subsub-in-measurement.csv"),
            "standard"
        )

        # Repeat static measuremnts in the temporal dimension as they do not
        # change, these are flattened in case the dataset is not sequential.
        # We store the amount of static features
        measurements_feats = static_msr_feats[None, :, :].repeat(
            self.lag, 1, 1
        )

        subsubwatersheds_feats = static_subsub_feats[None, :, :].repeat(
            self.lag, 1, 1
        )

        if self.sequential:
            # Edge indices are repeated across the temporal dimension only if
            # we are working with a sequential data as reflattening them is
            # not simple
            msr_flows_msr = msr_flows_msr.repeat(self.lag, 1, 1)
            sub_flows_sub = sub_flows_sub.repeat(self.lag, 1, 1)
            sub_in_msr = sub_in_msr.repeat(self.lag, 1, 1)

        self.scaler = get_scaler(self.scaler_name)

        # Scale all columns besides target and unscalable columns
        scaled_cols = [
            col for col in df_lagged if col not in [
                self.target_var, "date", "station_number"
            ]
        ]

        df_lagged[scaled_cols] = self.scaler.fit_transform(
            df_lagged[scaled_cols]
        )

        # Scale target column separately as we need to inverse transform later
        df_lagged[[self.target_var]] = self.scaler.fit_transform(
            df_lagged[[self.target_var]]
        )

        unique_dates = df_lagged.date.unique()

        for date in tqdm(unique_dates, desc="date"):
            date = np.datetime64(date, "D")
            df_date = df_lagged.loc[
                df_lagged.date == date
            ].sort_values("station_number")

            # Extract date features and add the static features
            date_flow_features = torch.from_numpy(
                df_date.loc[
                    :, df_date.columns.str.match(f"river_flow-\\d+")
                ].to_numpy()
            )

            # Extract flow as measurement features
            date_flow_features = date_flow_features.T.reshape(self.lag, 4, 1)
            msr_features = torch.cat(
                [measurements_feats, date_flow_features], dim=2
            )

            # Potentially extract ndsi and ndvi features, otherwise use static
            subsub_features = subsubwatersheds_feats

            if self.ndsi_features:
                df_date_NDSI = df_NDSI_lagged.loc[df_NDSI_lagged.date == date]
                date_NDSI_features = torch.from_numpy(
                    df_date_NDSI.loc[
                        :, df_date_NDSI.columns.str.fullmatch("NDSI.*\\d+")
                    ].to_numpy()
                )

                date_NDSI_features = date_NDSI_features.T.reshape(
                    self.lag, n_subsubs, -1
                )

                subsub_features = torch.cat(
                    (subsub_features, date_NDSI_features), dim=2
                )

            if self.ndvi_features:
                df_date_NDVI = df_NDVI_lagged.loc[df_NDVI_lagged.date == date]
                date_NDVI_features = torch.from_numpy(
                    df_date_NDVI.loc[
                        :, df_date_NDVI.columns.str.fullmatch("NDVI.*\\d+")
                    ].to_numpy()
                )

                date_NDVI_features = date_NDVI_features.T.reshape(
                    self.lag, n_subsubs, -1
                )

                subsub_features = torch.cat(
                    (subsub_features, date_NDVI_features), dim=2
                )

            # Flatten the lag and feature dimension
            if not self.sequential:
                temporal_msr_features = msr_features[
                    :, :, n_static_msr_feats:
                ].permute(1, 0, 2).flatten(1, 2)

                temporal_subsub_features = subsub_features[
                    :, :, n_static_subsub_feats:
                ].permute(1, 0, 2).flatten(1, 2)

                msr_features = torch.cat(
                    (static_msr_feats, temporal_msr_features), dim=1
                )
                subsub_features = torch.cat(
                    (static_subsub_feats, temporal_subsub_features), dim=1
                )

            # Extract date targets and convert to tensor
            df_date_targets = df_date.loc[
                df_date["station_number"].isin(self.target_stations)
            ]

            date_targets = torch.from_numpy(df_date_targets.loc[
                :, df_date_targets.columns.str.fullmatch("river_flow\\+\\d+")
            ].to_numpy())

            # We always want 6 predictions, so aggregate weekly into monthly
            if self.freq == "W":
                date_targets = date_targets.reshape((4, -1, 4)).mean(dim=2)

            if self.sequential:
                # Mapping defines our graph
                mapping = {
                    ("measurement", "flows", "measurement"): {
                        "edge_indices": msr_flows_msr
                    },
                    ("subsub", "flows", "subsub"): {
                        "edge_indices": sub_flows_sub
                    },
                    ("subsub", "in", "measurement"): {
                        "edge_indices": sub_in_msr
                    },
                    "measurement": {
                        "xs": msr_features.float(),
                        "y": date_targets.float()
                    },
                    "subsub": {"xs": subsub_features.float()}
                }

                data = HeteroSeqData(mapping, self.lag)
            else:
                mapping = {
                    ("measurement", "flows", "measurement"): {
                        "edge_index": msr_flows_msr
                    },
                    ("subsub", "flows", "subsub"): {
                        "edge_index": sub_flows_sub
                    },
                    ("subsub", "in", "measurement"): {
                        "edge_index": sub_in_msr
                    },
                    "measurement": {
                        "x": msr_features.float(),
                        "y": date_targets.float()
                    },
                    "subsub": {"x": subsub_features.float()}
                }

                data = HeteroData(mapping)

            self.data_date_dict[date] = data

    def set_data(self, date: np.datetime64, value: HeteroData) -> None:
        """
        class function: set_data

        Sets a certain value (or graph) on a certain date in the dataset
        """
        self.data_date_dict[date] = value

    def get_item_by_date(self, date: np.datetime64) -> HeteroData:
        """
        class function: get_item_by_date

        Returns graph based on the date provided
        """
        return self.data_date_dict[date]

    @property
    def data_list(self) -> List[Any]:
        assert self.data_date_dict
        return list(self.data_date_dict.values())

    def __repr__(self) -> str:
        return repr(self.data_list)

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index: int) -> HeteroData:
        return self.data_list[index]


def split_dataset(
    dataset: GraphFlowDataset, freq: str,
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
        offset = pd.tseries.offsets.DateOffset(months=dataset.lag)
    if freq == "W":
        offset = pd.tseries.offsets.DateOffset(weeks=dataset.lag)

    train_dataset = GraphFlowDataset()
    val_dataset = GraphFlowDataset()
    test_dataset = GraphFlowDataset()

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


def generate_lags(
    df: pd.DataFrame, values: List[str], n_lags: int, groupby_col: str
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
    for _, group in df.groupby(groupby_col):

        # Lag each dataframe individually
        df_n = group.copy()

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
    df_merged = pd.concat(frames)

    return df_merged


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
    df_flow = pd.read_csv(join(root, "temporal", "measurements.csv"),
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

    return df_flow_aggregated


def load_and_aggregate_ndsi_ndvi_data(
    root: Union[os.PathLike[str], str], name: str, freq: str = "W",
    index: bool = False, surface: bool = False, cloud: bool = False
) -> pd.DataFrame:
    """
    function: gather_ndsi_ndvi_data

    Gather ndsi ndvi data and returns whatever is indicated by boolean flags
    """

    df = gather_ndsi_ndvi_data(os.path.join(root, "temporal"), f"{name}.csv")

    subsub_dfs = []

    date_range = pd.date_range("1965", "2022", freq=freq, normalize=True)

    for subsub, group_df in df.groupby("Subsubwatershed"):
        df_subsub_aggr = pd.DataFrame(date_range, columns=["date"])

        if index:
            df_index = aggregate_index_data(name, freq, group_df)
            df_subsub_aggr = pd.merge(df_subsub_aggr, df_index, how="left")

        if surface:
            df_surface = aggregate_area_data(name, freq, group_df, "Surfavg")
            df_subsub_aggr = pd.merge(df_subsub_aggr, df_surface, how="left")

        if cloud:
            df_surface = aggregate_area_data(name, freq, group_df,
                                             "Surfcloudavg")
            df_subsub_aggr = pd.merge(df_subsub_aggr, df_surface, how="left")

        df_subsub_aggr["Subsubwatershed"] = subsub
        subsub_dfs.append(df_subsub_aggr)

    df_out = pd.concat(subsub_dfs)

    return df_out


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
    df = pd.read_csv(os.path.join(root, filename),
                     index_col=0, parse_dates=["date"],
                     dtype={"Subsubwatershed": int})

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

    grouped_df = df.groupby("date")[[column]].sum().reset_index()

    freq_surf_mean = grouped_df.groupby(
                                pd.Grouper(key='date', freq=freq)
                            ).mean().reset_index()

    freq_surf_mean = freq_surf_mean.rename({column: f"{name}_{column}"},
                                           axis="columns")

    return freq_surf_mean


def aggregate_index_data(
    name: str, freq: str, df: pd.DataFrame
) -> pd.DataFrame:
    """
    Returns the aggregated NDSI NDVI data with lagged variables
    Args:
        df_NDSI: dataframe containing filtered NDSI values
        df_NDVI: dataframe containing filtered NDVI values
    """

    # Take average of NDSI values for each month and aggregate
    freq_mean = df.groupby(pd.Grouper(key='date', freq=freq))[["avg"]].mean()
    freq_mean = freq_mean.reset_index()

    mean_df = freq_mean.rename({"avg": f"{name}_avg"}, axis="columns")

    return mean_df
