import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler


class RiverFlowDataset(Dataset):
    def __init__(self, df, scaler=None):

        x = df.loc[:, ~df.columns.isin(["river_flow", "date"])].values
        y = df.loc[:, df.columns.isin(["river_flow"])].values

        if scaler:
            x = scaler.fit_transform(x)
            y = scaler.fit_transform(y)

            self.scaler = scaler

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


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
    scalers = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "robust": RobustScaler
    }

    return scalers[scaler.lower()]()


def gather_river_flow_data(lag=6, time_features=False, index_features=False):
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

    # Convert dataset to lagged dataset
    df_features = generate_lags(flow_mean_df, ["river_flow"], lag)

    # Add time as feature if boolean is True
    if time_features:

        df_features = (
            df_features
            .assign(month=df_features.date.dt.month)
        )

        df_features = generate_cyclical_features(df_features, "month", 12, 1)

    if index_features:
        df_features = merge_flow_ndsi_ndvi_df(df_features, lag=lag)

    return df_features


def gather_ndsi_ndvi_data(lag=6):
    """
    This function returns the full processed data using various arguments
    as a pd.DataFrame
    Args:
        lag: amount of time lag to be used as features
    """
    processed_folder_path = os.path.join("data", "processed")

    df_NDSI = pd.read_csv(os.path.join(processed_folder_path, "NDSI.csv"),
                          index_col=0, parse_dates=["date"],
                          dtype={"Subsubwatershed": str})
    df_NDVI = pd.read_csv(os.path.join(processed_folder_path, "NDVI.csv"),
                          index_col=0, parse_dates=["date"],
                          dtype={"Subsubwatershed": str})

    # Only preserve rows inside subsubwatershed list
    watersheds = ["03400", "03401", "03402",
                  "03403", "03404", "03410",
                  "03411", "03412", "03413",
                  "03414", "03420", "03421"]

    keep_rows_ndsi = df_NDSI[df_NDSI.Subsubwatershed.isin(watersheds)].index
    keep_rows_ndvi = df_NDVI[df_NDVI.Subsubwatershed.isin(watersheds)].index

    df_NDSI = df_NDSI[df_NDSI.index.isin(keep_rows_ndsi)]
    df_NDVI = df_NDVI[df_NDVI.index.isin(keep_rows_ndvi)]

    # Take sum of each day and average over the months to aggregate area data
    daily_ndsi_surf_sum = df_NDSI.groupby(
                            pd.PeriodIndex(df_NDSI.date, freq="M")
                        )[["Surfavg"]].sum()
    daily_ndvi_surf_sum = df_NDVI.groupby(
                            pd.PeriodIndex(df_NDVI.date, freq="M")
                        )[["Surfavg"]].sum()

    monthly_ndsi_surf_mean = daily_ndsi_surf_sum.groupby(pd.PeriodIndex(
                                daily_ndsi_surf_sum.index, freq="M")
                            )[["Surfavg"]].mean()
    monthly_ndvi_surf_mean = daily_ndvi_surf_sum.groupby(pd.PeriodIndex(
                                daily_ndvi_surf_sum.index, freq="M")
                            )[["Surfavg"]].mean()

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
    surf_ndsi_mean_df = monthly_ndsi_surf_mean.reset_index()
    surf_ndvi_mean_df = monthly_ndvi_surf_mean.reset_index()

    ndsi_mean_df = ndsi_mean_df.rename({"avg": "ndsi_avg"}, axis="columns")
    ndvi_mean_df = ndvi_mean_df.rename({"avg": "ndvi_avg"}, axis="columns")
    surf_ndsi_mean_df = surf_ndsi_mean_df.rename({"Surfavg": "ndsi_surf_avg"},
                                                 axis="columns")
    surf_ndvi_mean_df = surf_ndvi_mean_df.rename({"Surfavg": "ndvi_surf_avg"},
                                                 axis="columns")

    # Merge ndvi and ndsi dataframes into one
    ndsi_ndvi_mean_df = pd.merge(ndsi_mean_df, ndvi_mean_df)
    ndsi_ndvi_mean_df = pd.merge(ndsi_ndvi_mean_df, surf_ndsi_mean_df)
    ndsi_ndvi_mean_df = pd.merge(ndsi_ndvi_mean_df, surf_ndvi_mean_df)

    ndsi_ndvi_mean_df = generate_lags(ndsi_ndvi_mean_df,
                                      ["ndsi_avg", "ndvi_avg",
                                       "ndsi_surf_avg", "ndvi_surf_avg"],
                                      lag)

    ndsi_ndvi_mean_df = ndsi_ndvi_mean_df.drop(
                            columns=["ndsi_avg", "ndvi_avg",
                                     "ndsi_surf_avg", "ndvi_surf_avg"]
                        )

    return ndsi_ndvi_mean_df


def merge_flow_ndsi_ndvi_df(df_features, lag=6):
    """
    The ndsi_ndvi_mean_df will be merged into df_features and the
    full dataframe is returned.
    Args:
        df_features: dataframe to merge ndsi_ndvi_df into
    """
    df_ndsi_ndvi = gather_ndsi_ndvi_data(lag=lag)

    df_features = pd.merge(df_features, df_ndsi_ndvi, how="left")
    df_features = df_features.dropna(subset=["river_flow"])
    df_features = df_features.fillna(-1, downcast="infer")

    return df_features


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
