from datetime import datetime
from matplotlib import pyplot as plt

import torch
import numpy as np
import pandas as pd
from pandas import DataFrame


def split_data(
    date_train_end: datetime, date_val_end: datetime, weather_data: pd.DataFrame
) -> tuple[DataFrame, DataFrame, DataFrame]:
    train_data = weather_data[weather_data["Date"] <= date_train_end]
    val_data = weather_data[
        (weather_data["Date"] > date_train_end) & (weather_data["Date"] <= date_val_end)
    ]
    test_data = weather_data[(weather_data["Date"] > date_val_end)]
    return train_data, val_data, test_data


def process_data(
    weather_data: pd.DataFrame, hourly_intervals=True
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        # create label column
        weather_data["next_T"] = weather_data["T"].shift(-1)
    label_temperature = weather_data["next_T"]
    # encode cyclic data
    weather_data = encode_cyclic_data(weather_data)
    # delete unnecessary columns
    weather_data = clean_dataframe(weather_data)
    # reorganize columns
    weather_data = weather_data[
        [
            "sin_day_of_year",
            "cos_day_of_year",
            "sin_hour",
            "cos_hour",
            "P0",
            "T",
            "U",
            "Ff",
        ]
    ]

    return weather_data, label_temperature


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
    return weather_data


def get_hourly_data(weather_data: pd.DataFrame) -> pd.DataFrame:
    hourly_weather = weather_data[weather_data["minutes"] == 0]
    return hourly_weather


def clean_dataframe(weather_data: pd.DataFrame) -> pd.DataFrame:
    weather_data.drop(
        ["Date", "time", "minutes", "seconds", "hour", "day_of_year"],
        axis=1,
        inplace=True,
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


def create_plot(
    epoch_train_losses,
    epoch_val_losses,
    epoch_test_losses,
    name,
    e,
    save_plot=True,
    display_plot=False,
):
    figure, axis = plt.subplots(2, 1)
    # subplot 1
    axis[0].plot(epoch_train_losses, c="r")
    axis[0].plot(epoch_val_losses, c="b")
    axis[0].legend(["Train_loss", "Val_loss"])
    axis[0].set_title("Train vs. Validation loss")
    axis[0].set(xlabel="Epoch", ylabel="Loss")
    # subplot 2
    axis[1].plot(epoch_test_losses, c="g")
    axis[1].legend(["Test_loss"])
    axis[1].set_title("Test loss")
    axis[1].set(xlabel="Epoch", ylabel="Loss")
    plt.tight_layout()
    if save_plot:
        plt.savefig(f"models/plots/WP-{e}-{name}.png")
    if display_plot:
        plt.show()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.has_mps:
        return "mps"
    else:
        return "cpu"
