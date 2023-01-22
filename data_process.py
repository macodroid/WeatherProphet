import pandas as pd


def process_data(file_location: str, hourly_intervals=True) -> pd.DataFrame:
    weather_data = pd.read_csv(file_location, parse_dates=["Date"])

    if weather_data.isnull().values.any():
        # delete rows with missing values
        weather_data.dropna(axis=0, inplace=True)
        # reset index
        weather_data.reset_index(drop=True, inplace=True)
    # create a new column with day of the year
    weather_data = add_day_of_year(weather_data)
    if hourly_intervals:
        weather_data = get_hourly_data(weather_data)
    # delete unnecessary columns
    weather_data = clean_dataframe(weather_data)

    return weather_data


def add_day_of_year(weather_data: pd.DataFrame) -> pd.DataFrame:
    weather_data["DayOfYear"] = weather_data["Date"].dt.dayofyear
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
        ["Date", "time", "minutes", "seconds", 'P'], axis=1, inplace=True
    )
    return weather_data