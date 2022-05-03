import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from typing import Type, Optional, Tuple, List


def generate_lags(df: pd.DataFrame, values: list, n_lags: int) -> pd.DataFrame:
    """
    function: generate_lags

    Generates a dataframe with columns denoting lagged value up to n_lags
    """
    df_n = df.copy()

    for value in values:
        for n in range(1, n_lags + 1):
            df_n[f"{value}_{n}"] = df_n[f"{value}"].shift(n)

    return df_n


def get_scaler(scaler: str) -> Optional[Type]:
    """
    function: get_scaler

    Returns a scaler from a selection of 4 options, given the string name.
    """
    if scaler == "none":
        return None

    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }

    return scalers[scaler.lower()]()


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


def load_edges() -> Tuple[dict, dict]:
    """
    Load the edges from disk, and convert to the format specified by
    PyTorch Geometric Temporal. We return an edge and edge feature dictionary
    where the keys are the specific type of edge and the value the
    corresponding edges and features.
    """
    processed_path = os.path.join("data", "processed")

    # Load edge data from disk
    edges_msrmsr = pd.read_csv(os.path.join(processed_path,
                               "measurement-flows-measurement.csv"),
                               index_col=0)
    edges_subsub = pd.read_csv(os.path.join(processed_path,
                               "subsub-flows-subsub.csv"),
                               index_col=0)
    edges_submsr = pd.read_csv(os.path.join(processed_path,
                               "subsub-in-measurement.csv"),
                               index_col=0)

    # Convert edges to proper format
    edges_msrmsr_arr = edges_msrmsr[["src", "dst"]].to_numpy().T
    edges_subsub_arr = edges_subsub[["src", "dst"]].to_numpy().T
    edges_submsr_arr = edges_submsr[["src", "dst"]].to_numpy().T

    # Gather edge features (distance in our case)
    edge_msrmsr_feats = edges_msrmsr.distance.to_numpy()
    edge_subsub_feats = edges_subsub.distance.to_numpy()
    edge_submsr_feats = np.ones(len(edges_submsr))

    # Create the dictionaries of the various types of edges
    edges_dict = {
        ("measurement", "flows", "measurement"): edges_msrmsr_arr,
        ("sub", "flows", "sub"): edges_subsub_arr,
        ("sub", "in", "measurement"): edges_submsr_arr
    }

    edges_feats_dict = {
        ("measurement", "flows", "measurement"): edge_msrmsr_feats,
        ("sub", "flows", "sub"): edge_subsub_feats,
        ("sub", "in", "measurement"): edge_submsr_feats
    }

    return edges_dict, edges_feats_dict


def load_nodes(target_variable: str = "river_flow",
               lagged_variables: List[str] = ["river_flow"],
               freq: str = "M",
               lag: int = 6) -> Tuple[List[dict], List[dict]]:
    """
    function: load_nodes

    Loads the nodes from disks, first aggregates the measurements then returns
    the data in the form of two lists of dictionaries with an item for each
    date in the requested dataset. The period of these dates is defined by
    freq.
    """

    df_flow_aggregated = load_and_aggregate_flow_data(freq=freq)

    processed_path = os.path.join("data", "processed")

    # Lagging in separate dataframes negates overflow between station nubmers
    station_dfs = []

    for _, df_station_flow_agg in df_flow_aggregated.groupby("station_number"):
        station_dfs.append(
            generate_lags(df_station_flow_agg,
                          lagged_variables, lag)
            )

    df_flow_agg_lagged = pd.concat(station_dfs)

    # Load static features of data
    nodes_sub = pd.read_csv(os.path.join(processed_path, "subsub.csv"),
                            index_col=0)
    nodes_sub_feats = nodes_sub.to_numpy()

    nodes_msr = pd.read_csv(os.path.join(processed_path, "measurement.csv"),
                            index_col=0)

    nodes_msr_static_feats = nodes_msr.to_numpy()

    unique_dates = df_flow_aggregated.date.unique()

    target_dicts = []
    feature_dicts = []

    for date in unique_dates:
        date_df = df_flow_agg_lagged.loc[df_flow_agg_lagged.date == date]
        date_df = date_df.sort_values("station_number")

        feature_df = date_df.loc[:, date_df.columns.str.match(".*_\\d")]

        date_features = feature_df.to_numpy()

        feature_dicts.append({
                "measurement": date_features,
                "sub": nodes_sub_feats
            })

        target_dicts.append({
                "measurement": date_df[target_variable].values
            })

    return feature_dicts, target_dicts
