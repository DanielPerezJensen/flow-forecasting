def generate_lags(df, value, n_lags):
    """
    generate_lags
    Generates a dataframe with columns denoting lagged value up to n_lags
    Args:
        df: dataframe to lag
        value: value to lag
        n_lags: amount of rows to lag
    """
    df_n = df.copy()

    for n in range(1, n_lags + 1):
        df_n[f"lag_{n}"] = df_n[f"{value}"].shift(n)

    df_n = df_n.iloc[n_lags:]

    return df_n