import pandas as pd


def video_to_picture_ratio(df, videos_col, pictures_col):
    df['video_picture_ratio'] = df[videos_col] / df[pictures_col]
    return df


def calculate_rolling_average_per_brand(data, brand_column, window_size=7):
    """
    Calculate the rolling average of 'engagement' for each brand in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        brand_column (str): The column name which identifies the brand.
        window_size (int): The number of observations used for calculating the rolling average.

    Returns:
        pd.DataFrame: The DataFrame with a new column for the rolling average of 'engagement' calculated per brand.
    """
    # Group by brand and calculate rolling average within each group
    data['rolling_avg_engagement'] = data.groupby(brand_column)['engagement'].transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
    return data


def calculate_exponential_moving_average_per_brand(data, brand_column, span=7):
    """
    Calculate the exponential moving average of 'engagement' for each brand in a DataFrame.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        brand_column (str): The column name which identifies the brand.
        span (int): The decay in terms of the span of the exponential window.

    Returns:
        pd.DataFrame: The DataFrame with a new column for the exponential moving average of 'engagement' calculated per brand.
    """
    # Group by brand and calculate EMA within each group
    data['ewma_engagement'] = data.groupby(brand_column)['engagement'].transform(lambda x: x.ewm(span=span, adjust=False).mean())
    return data


def calculate_brand_wise_growth_rates(data, column_names, brand_column):
    """
    Calculate the growth rate for specified columns in a DataFrame, grouped by brand.
    Replace NaN values with 0 where growth rate cannot be calculated.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_names (list of str): A list of column names to calculate the growth rate.
        brand_column (str): The column name which identifies the brand.

    Returns:
        pd.DataFrame: The DataFrame with new columns for each specified column's growth rate, calculated for each brand and NaN replaced by 0.
    """
    for column in column_names:
        # Calculate growth rate within each brand group
        data[f'growth_rate_{column}'] = data.groupby(brand_column)[column].pct_change() * 100
        # Replace NaN values with 0
        data[f'growth_rate_{column}'].fillna(0, inplace=True)
    return data

def calculate_brand_rolling_statistics(data, column_name, brand_column, window_size=7):
    """
    Calculate rolling statistics for a specified column in a DataFrame, grouped by brand,
    and replace NaN values with 0.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column name to calculate rolling statistics.
        brand_column (str): The column name which identifies the brand.
        window_size (int): The number of observations used for calculating the rolling statistic.

    Returns:
        pd.DataFrame: The DataFrame with new columns for each rolling statistic of the specified column, calculated for each brand.
    """
    grouped = data.groupby(brand_column)[column_name]
    data[f'{column_name}_rolling_min'] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).min()).fillna(0)
    data[f'{column_name}_rolling_max'] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).max()).fillna(0)
    data[f'{column_name}_rolling_std'] = grouped.transform(lambda x: x.rolling(window=window_size, min_periods=1).std()).fillna(0)

    return data

def create_brand_lag_features(data, column_name, brand_column, lag_periods):
    """
    Create lag features for a specified column in a DataFrame, grouped by brand,
    and replace NaN values with 0.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The column name to create lag features for.
        brand_column (str): The column name which identifies the brand.
        lag_periods (int): The number of lag periods.

    Returns:
        pd.DataFrame: The DataFrame with new columns for each lag feature of the specified column, calculated for each brand.
    """
    for i in range(1, lag_periods + 1):
        data[f'{column_name}_lag_{i}'] = data.groupby(brand_column)[column_name].shift(i).fillna(0)

    return data