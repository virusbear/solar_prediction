from datetime import timedelta, datetime, timezone
from prom import prom_query_range
from functools import reduce
import pandas as pd
import numpy as np
from astral.sun import azimuth, elevation
from astral import Observer

PROMETHEUS_URL = 'http://monitoring.local:8481/select/0/prometheus'
COLUMN_LABEL = 'friendly_name'

def _load_metric(metric, start: datetime = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc), end: datetime = datetime.now(timezone.utc), interval: timedelta = timedelta(minutes=5)):
    return prom_query_range(
        PROMETHEUS_URL,
        COLUMN_LABEL,
        metric,
        start=start,
        end=end,
        step=interval
    )

def _enrich_time_metrics(df):
    timestamp_s = df.index.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    year = 365.2425 * day
    df['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

def _enrich_solar_metrics(df, latitude, longitude, height):
    observer = Observer(latitude, longitude, height)
    df['elevation'] = [elevation(observer, timestamp) for timestamp in df.index]
    df['azimuth'] = [azimuth(observer, timestamp) for timestamp in df.index]

def _rename_columns(df):
    df.rename(columns={
        'Solarbank 2 E1600 Pro Solarleistung': 'yield',
        'Solarbank 2 E1600 Pro Ladestand': 'level',
        'Sun Solare Elevation': 'elevation',
        'Sun Solarer Azimut': 'azimuth'
    }, inplace=True)

def load_dataset(latitude: float, longitude: float, elevation: float, start: datetime = datetime(year=1970, month=1, day=1, tzinfo=timezone.utc), end: datetime = datetime.now(timezone.utc), interval: timedelta = timedelta(minutes=5)) -> pd.DataFrame:
    solar_yield = _load_metric('ha_sensor_power_w{entity="sensor.solarbank_2_e1600_pro_solarleistung", retention="inf"}', start, end, interval)
    battery_percentage = _load_metric('ha_sensor_battery_percent{entity="sensor.solarbank_2_e1600_pro_ladestand",retention="inf"}', start, end, interval)
    #elevation = _load_metric('ha_sensor_unit_u0xb0{entity="sensor.sun_solar_elevation",retention="inf"}', start, end, interval)
    #azimuth = _load_metric('sin(ha_sensor_unit_u0xb0{entity="sensor.sun_solar_azimuth",retention="inf"} * pi() / 180)', start, end, interval)

    def merge_data_frame(frames):
        return reduce(lambda x, y: pd.merge_asof(x, y, on='timestamp', direction='nearest'), frames)

    #df = merge_data_frame([solar_yield, battery_percentage, elevation, azimuth])
    df = merge_data_frame([solar_yield, battery_percentage])
    df.set_index('timestamp', inplace=True)
    index_start = pd.date_range(start=df.index.min(), end=start.astimezone(timezone.utc), freq=-interval).min()
    index = pd.date_range(start=index_start, end=end, freq=interval)
    df = df.reindex(index)
    _enrich_time_metrics(df)
    _enrich_solar_metrics(df, latitude, longitude, elevation)
    _rename_columns(df)
    return df