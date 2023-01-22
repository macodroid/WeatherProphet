from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame


def split_data(date_train_end: datetime,
               date_val_end: datetime,
               weather_data: pd.DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    train_data = weather_data[weather_data['Date'] <= date_train_end]
    val_data = weather_data[(weather_data['Date'] > date_train_end) & (weather_data['Date'] <= date_val_end)]
    test_data = weather_data[(weather_data['Date'] > date_val_end)]
    return train_data, val_data, test_data


def process_data(weather_data: pd.DataFrame, hourly_intervals=True) -> pd.DataFrame:
    if weather_data.isnull().values.any():
        # delete rows with missing values
        weather_data.dropna(axis=0, inplace=True)
        # reset index
        weather_data.reset_index(drop=True, inplace=True)
    # create a new column with day of the year
    weather_data = add_day_of_year(weather_data)
    weather_data = add_time_columns(weather_data)
    if hourly_intervals:
        weather_data = get_hourly_data(weather_data)
    # encode cyclic data
    weather_data = encode_cyclic_data(weather_data)
    # delete unnecessary columns
    weather_data = clean_dataframe(weather_data)
    # reorganize columns
    weather_data = weather_data[['sin_day_of_year', 'cos_day_of_year', 'sin_hour', 'cos_hour', 'P0', 'U', 'Ff', 'T']]

    return weather_data


def add_day_of_year(weather_data: pd.DataFrame) -> pd.DataFrame:
    weather_data["day_of_year"] = weather_data["Date"].dt.dayofyear
    return weather_data


def add_time_columns(weather_data: pd.DataFrame) -> pd.DataFrame:
    weather_data["time"] = pd.to_datetime(
        weather_data["Date"], format="%H:%M:%S"
    ).dt.time
    weather_data["time"] = weather_data["time"].astype("string")
    weather_data[["hour", "minutes", "seconds"]] = weather_data["time"].str.split(
        ":", expand=True
    )
    weather_data[["hour", "minutes", "seconds"]] = weather_data[
        ["hour", "minutes", "seconds"]
    ].astype(int)
    # weather_data['minutes'] = weather_data['minutes'].astype(int)
    # weather_data['seconds'] = weather_data['seconds'].astype(int)
    return weather_data


def get_hourly_data(weather_data: pd.DataFrame) -> pd.DataFrame:
    hourly_weather = weather_data[weather_data["minutes"] == 0]
    return hourly_weather


def clean_dataframe(weather_data: pd.DataFrame) -> pd.DataFrame:
    weather_data.drop(
        ["Date", "time", "minutes", "seconds", 'P', 'hour', 'day_of_year'], axis=1, inplace=True
    )
    return weather_data


def encode_cyclic_data(weather_data: pd.DataFrame) -> pd.DataFrame:
    weather_data["sin_day_of_year"] = np.sin(
        2 * np.pi * weather_data["day_of_year"] / 365
    )
    weather_data["cos_day_of_year"] = np.cos(
        2 * np.pi * weather_data["day_of_year"] / 365
    )
    weather_data["sin_hour"] = np.sin(2 * np.pi * weather_data["hour"] / 24)
    weather_data["cos_hour"] = np.cos(2 * np.pi * weather_data["hour"] / 24)
    return weather_data


def normalize_data(weather_data: pd.DataFrame, mean: float = None, std: float = None):
    if (mean is None) and (std is None):
        mean = weather_data.mean()
        weather_data -= mean
        std = weather_data.std()
        weather_data /= std
        return weather_data, mean, std
    weather_data -= mean
    weather_data /= std
    return weather_data
