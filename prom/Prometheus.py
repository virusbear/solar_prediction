from prometheus_api_client import PrometheusConnect
from datetime import datetime, time, timedelta, date, timezone
import pandas as pd
import numpy as np

def _metric_to_dataframe(metric, col_label_name):
    series = np.array([np.array(x) for x in metric['values']]).T
    return pd.DataFrame({
        'timestamp': pd.to_datetime(pd.to_numeric(series[0]), unit="s", utc=True),
        metric['metric'][col_label_name]: series[1].astype(float),
    }).sort_values('timestamp')

def _merge_metric_dataframes(df, metric):
    _rename_duplicate_column(df, metric)
    return pd.merge_asof(df, metric, on='timestamp', direction='nearest')

def _rename_duplicate_column(df, metric):
    for i in range(len(metric.columns)):
        name = metric.columns[i]
        basename = name
        x = 0
        while _column_exists(df, metric.columns[i]):
            new_name = '{}_({})'.format(basename, x)
            metric.rename(columns={name: new_name}, inplace=True)
            name = new_name
            x += 1

def _column_exists(df, column):
    return column in df.columns and column != 'timestamp'

def _prom_query_range(url: str, col_label_name: str, query: str, start: datetime, end: datetime, step: timedelta) -> pd.DataFrame:
    prom = PrometheusConnect(url)
    results = prom.custom_query_range(query, start_time=start.astimezone(timezone.utc), end_time=end.astimezone(timezone.utc), step=str(int(step.total_seconds())) + 's')
    dataframes = [_metric_to_dataframe(result, col_label_name) for result in results]
    if len(dataframes) == 0:
        return pd.DataFrame()

    df = dataframes[0]
    for metric in dataframes[1:]:
        df = _merge_metric_dataframes(df, metric)

    return df

def prom_query_range(url: str, col_label_name: str, query: str, start: datetime, end: datetime, step: timedelta, chunk_size: int = 20000) -> pd.DataFrame:
    if (end - start) / step < chunk_size:
        return _prom_query_range(url, col_label_name, query, start, end, step)
    else:
        end_time = start - step
        df = pd.DataFrame()

        while end_time < end:
            start_time = end_time + step
            end_time = min(end, start_time + (step * (chunk_size - 1)))
            result = _prom_query_range(url, col_label_name, query, start_time, end_time, step)
            df = pd.concat([df, result])

        df.set_index('timestamp', inplace=True)
        return df