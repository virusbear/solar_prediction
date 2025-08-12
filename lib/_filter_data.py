import numpy as np
import pandas as pd
from datetime import timedelta


def _remove_battery_full_samples(df: pd.DataFrame) -> pd.DataFrame:
    full = df['level'] > 99
    df[full, 'yield'] = np.nan
    df = df.drop(['level'], axis=1)
    return df

def _remove_solar_idle_samples(df: pd.DataFrame, idle_samples: int) -> pd.DataFrame:
    return df[df['yield'].rolling(idle_samples).sum() != 0]

def _split_samples(df: pd.DataFrame, normal_sample_interval: timedelta) -> [pd.DataFrame]:
    df = df.sort_index()
    time_diffs = df.index.to_series().diff()
    delta = pd.Timedelta(normal_sample_interval)
    split_points = np.where(time_diffs > delta)[0]
    result = []
    start = 0

    for idx in split_points:
        result.append(df.iloc[start:idx])
        start = idx

    result.append(df.iloc[start:])

    return result

def _filter_short_samples(dfs: [pd.DataFrame], min_data_duration: timedelta) -> [pd.DataFrame]:
    result = []

    for d in dfs:
        if d.index.max() - d.index.min() > min_data_duration:
            result.append(d)

    return result

def filter_data(df: pd.DataFrame, idle_samples: int = 12, normal_sample_interval: timedelta = timedelta(minutes=5), min_data_duration: timedelta = timedelta(hours=4)) -> [pd.DataFrame]:
    #df = _remove_solar_idle_samples(df, idle_samples)
    df = _remove_battery_full_samples(df)
    return _filter_short_samples(_split_samples(df, normal_sample_interval), min_data_duration)
