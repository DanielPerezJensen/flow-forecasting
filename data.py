import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


class RiverFlowDataset(Dataset):
    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_lags(df, values, n_lags):
    """
    generate_lags
    Generates a dataframe with columns denoting lagged value up to n_lags
    Args:
        df: dataframe to lag
        value: values to lag
        n_lags: amount of rows to lag
    """
    df_n = df.copy()

    for value in values:
        for n in range(1, n_lags + 1):
            df_n[f"{value}_{n}"] = df_n[f"{value}"].shift(n)

    df_n = df_n.iloc[n_lags:]

    return df_n


def generate_cyclical_features(df, col_name, period, start_num=0):

    kwargs = {
        f"sin_{col_name}":
            lambda x: np.sin(2 * np.pi * (df[col_name] - start_num) / period),
        f"cos_{col_name}":
            lambda x: np.cos(2 * np.pi * (df[col_name] - start_num) / period)
    }

    return df.assign(**kwargs).drop(columns=[col_name])


def get_scaler(scaler):
    if scaler == "none":
        return None

    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler,
    }

    return scalers[scaler.lower()]()


def gather_ndsi_ndvi_data(watersheds=None):
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


def aggregate_area_data(df_NDSI, df_NDVI, column):
    """
    This function will correctly aggregate area data given the column
    Args:
        df_NDSI: dataframe containing filtered NDSI values
        df_NDVI: dataframe containing filtered NDVI values
        column: column name to aggregate, must contain 'Surf'
    """

    if "Surf" not in column:
        raise InputError("'Surf' must be found in column name")

    # Take sum of each day and average over the months to aggregate area data
    daily_ndsi_surf_sum = df_NDSI.groupby(
                            pd.PeriodIndex(df_NDSI.date, freq="D")
                        )[[column]].sum()
    daily_ndvi_surf_sum = df_NDVI.groupby(
                            pd.PeriodIndex(df_NDVI.date, freq="D")
                        )[[column]].sum()

    monthly_ndsi_surf_mean = daily_ndsi_surf_sum.groupby(pd.PeriodIndex(
                                daily_ndsi_surf_sum.index, freq="M")
                            )[[column]].mean()
    monthly_ndvi_surf_mean = daily_ndvi_surf_sum.groupby(pd.PeriodIndex(
                                daily_ndvi_surf_sum.index, freq="M")
                            )[[column]].mean()

    surf_ndsi_mean_df = monthly_ndsi_surf_mean.reset_index()
    surf_ndvi_mean_df = monthly_ndvi_surf_mean.reset_index()
    surf_ndsi_mean_df = surf_ndsi_mean_df.rename({column: f"ndsi_{column}"},
                                                 axis="columns")
    surf_ndvi_mean_df = surf_ndvi_mean_df.rename({column: f"ndvi_{column}"},
                                                 axis="columns")

    surf_ndsi_ndvi_df = pd.merge(surf_ndsi_mean_df, surf_ndvi_mean_df)

    return surf_ndsi_ndvi_df


def aggregate_index_data(df_NDSI, df_NDVI):
    """
    Returns the aggregated NDSI NDVI data with lagged variables
    Args:
        df_NDSI: dataframe containing filtered NDSI values
        df_NDVI: dataframe containing filtered NDVI values
    """

    # Take average of NDSI values for each month and aggregate
    monthly_ndsi_mean = df_NDSI.groupby(pd.PeriodIndex(
                            df_NDSI.date, freq="M")
                        )[["avg"]].mean()
    monthly_ndvi_mean = df_NDVI.groupby(pd.PeriodIndex(
                            df_NDVI.date, freq="M")
                        )[["avg"]].mean()

    # Rename columns to enable merging
    ndsi_mean_df = monthly_ndsi_mean.reset_index()
    ndvi_mean_df = monthly_ndvi_mean.reset_index()

    ndsi_mean_df = ndsi_mean_df.rename({"avg": "ndsi_avg"}, axis="columns")
    ndvi_mean_df = ndvi_mean_df.rename({"avg": "ndvi_avg"}, axis="columns")

    # Merge ndvi and ndsi dataframes into one
    ndsi_ndvi_df = pd.merge(ndsi_mean_df, ndvi_mean_df)

    return ndsi_ndvi_df


def merge_aggregated_data(df_features, index=False,
                          surface=False, cloud=False):
    """
    We merge NDSI NDVI data into df_features and the
    Args:
        df_features: dataframe to merge ndsi_ndvi_df into
    """
    watersheds = ["03400", "03401", "03402",
                  "03403", "03404", "03410",
                  "03411", "03412", "03413",
                  "03414", "03420", "03421"]

    df_ndsi, df_ndvi = gather_ndsi_ndvi_data(watersheds=watersheds)

    if index:
        index_df = aggregate_index_data(df_ndsi, df_ndvi)
        df_features = pd.merge(df_features, index_df, how="left")

    if surface:
        surface_df = aggregate_area_data(df_ndsi, df_ndvi, "Surfavg")
        df_features = pd.merge(df_features, surface_df, how="left")

    if cloud:
        cloud_df = aggregate_area_data(df_ndsi, df_ndvi, "Surfcloudavg")
        df_features = pd.merge(df_features, cloud_df, how="left")

    df_features = df_features.dropna(subset=["river_flow"])
    df_features = df_features.fillna(-1, downcast="infer")

    return df_features


def gather_data(lag=6, time_features=False, index_features=False,
                index_surf_features=False, index_cloud_features=False):
    """
    This function returns the full preprocessed data using various arguments
    as a pd.DataFrame

    Args:
        int lag: amount of time lag to be used as features
        bool time_features: use (cyclical) time as a feature
    """
    processed_folder_path = os.path.join("data", "processed")

    # Import river flow data and only preserve datapoints after 1965
    df_DGA = pd.read_csv(os.path.join(processed_folder_path, "DGA.csv"),
                         index_col=0, parse_dates=["date"])
    df_DGA = df_DGA.loc[df_DGA["date"].dt.year >= 1965]

    # Extract average monthly river flow
    monthly_flow_data_mean = df_DGA.groupby(
                                    pd.PeriodIndex(df_DGA['date'], freq="M")
                                )['river_flow'].mean()
    flow_mean_df = monthly_flow_data_mean.reset_index()

    df_features = generate_lags(flow_mean_df, ["river_flow"], lag)

    # Add time as a feature
    if time_features:
        df_features = (
            df_features
            .assign(month=df_features.date.dt.month)
        )

        df_features = generate_cyclical_features(df_features, "month", 12, 1)

    # Add the index features if requested
    df_features = merge_aggregated_data(df_features, index=index_features,
                                        surface=index_surf_features,
                                        cloud=index_cloud_features)

    feat_cols = []

    if index_features:
        feat_cols += ["ndsi_avg", "ndvi_avg"]
    if index_surf_features:
        feat_cols += ["ndsi_Surfavg", "ndvi_Surfavg"]
    if index_cloud_features:
        feat_cols += ["ndsi_Surfcloudavg", "ndvi_Surfcloudavg"]

    df_features = generate_lags(df_features, feat_cols, lag)
    df_features = df_features.drop(columns=feat_cols)

    return df_features


def feature_label_split(df, target_col):
    """
    Split dataframe into two based on target_col
    """
    y = df[[target_col]]
    X = df.drop(columns=[target_col])

    return X, y


def split_data(df_features, lag,
               val_year_min=1998, val_year_max=2002,
               test_year_min=2013, test_year_max=2019):

    """
    This functions splits the data into a training, validation and test set
    based on the years and the lag provided
    Args:
        df_features: dataframe containing the features of the overall dataset
        lag: lag in dataset
        val_years: years to be used in validation set
        test_years: years to be used in test set
    """
    # We create an offset period range off year/months we don't want in the
    # training set
    offset = pd.tseries.offsets.DateOffset(months=lag)
    val_start = pd.to_datetime(str(val_year_min)) - offset
    test_start = pd.to_datetime(str(test_year_min)) - offset
    val_end = pd.to_datetime(str(val_year_max)) + offset
    test_end = pd.to_datetime(str(test_year_max)) + offset

    val_idx_offset = pd.period_range(start=val_start, end=val_end, freq="M")
    test_idx_offset = pd.period_range(start=test_start, end=test_end, freq="M")

    train_df = df_features.loc[~df_features.date.isin(val_idx_offset)]
    train_df = train_df.loc[~df_features.date.isin(test_idx_offset)]

    # The offset months should not be in the test and validation set, only the
    # ones we denote in the function
    real_val_idx = pd.period_range(start=val_year_min, end=val_year_max,
                                   freq="M")
    real_test_idx = pd.period_range(start=test_year_min, end=test_year_max,
                                    freq="M")

    val_df = df_features.loc[df_features.date.isin(real_val_idx)]
    test_df = df_features.loc[df_features.date.isin(real_test_idx)]

    return train_df, val_df, test_df


def scale_data(scaler, train_df, val_df, test_df):
    """
    Returns transformed data according to scaler provided,
    will return values as is if scaler is None
    """
    train_df = train_df.drop(columns=["date"])
    val_df = val_df.drop(columns=["date"])
    test_df = test_df.drop(columns=["date"])

    X_train, y_train = feature_label_split(train_df, "river_flow")
    X_val, y_val = feature_label_split(val_df, "river_flow")
    X_test, y_test = feature_label_split(test_df, "river_flow")

    # Transform the input according to chosen scaler if it is given
    if scaler is not None:
        X_train_arr = scaler.fit_transform(X_train.values)
        X_val_arr = scaler.transform(X_val.values)
        X_test_arr = scaler.transform(X_test.values)

        y_train_arr = scaler.fit_transform(y_train.values)
        y_val_arr = scaler.transform(y_val.values)
        y_test_arr = scaler.transform(y_test.values)
    else:
        X_train_arr = X_train.values
        X_val_arr = X_val.values
        X_test_arr = X_test.values

        y_train_arr = y_train.values
        y_val_arr = y_val.values
        y_test_arr = y_test.values

    return (X_train_arr, X_val_arr, X_test_arr,
            y_train_arr, y_val_arr, y_test_arr)
