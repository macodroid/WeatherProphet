import datetime

import numpy as np
import pandas as pd

import data_utils as helper

if __name__ == "__main__":
    weather_data_path = "dataset/BA_weather_dataset_manually_pre_processed.csv"
    # load data
    weather_data = pd.read_csv(weather_data_path, parse_dates=["Date"])
    # reverse order of rows
    weather_data = weather_data.iloc[::-1]

    # split dataset into train, validation and test.
    # The train dataset will start from 2012-10-01 (yyyy-mm-dd) and end at 2020-12-31.
    # The validation dataset will start from 2021-01-01 and end at 2021-12-31.
    # The test dataset will start from 2022-01-01 and end at till the end of the dataset.
    # (this depends on the freshness of dataset)
    # Leaving one whole year for validation and one whole year for testing because of the data seasonality.
    train_date_end = datetime.datetime(2020, 12, 31, 23, 59, 59)
    val_date_end = datetime.datetime(2021, 12, 31, 23, 59, 59)
    train_data, val_data, test_data = helper.split_data(
        train_date_end, val_date_end, weather_data
    )

    train_data = helper.process_data(train_data)
    val_data = helper.process_data(val_data)
    test_data = helper.process_data(test_data)
    # normalize data
    # Just for future. Inverse normalization X_unscaled = (X_scaled * std) + mean
    train_data, mean, std = helper.normalize_data(train_data)
    val_data = helper.normalize_data(val_data, mean, std)
    test_data = helper.normalize_data(test_data, mean, std)
    # convert to numpy array
    train_data, val_data, test_data = (
        train_data.to_numpy(),
        val_data.to_numpy(),
        test_data.to_numpy(),
    )
    np.save("dataset/train_data.npy", train_data)
    np.save("dataset/val_data.npy", val_data)
    np.save("dataset/test_data.npy", test_data)
    np.save("dataset/stat.npy", np.array([mean.to_numpy(), std.to_numpy()]))
