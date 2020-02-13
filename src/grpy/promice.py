import numpy as np
import pandas as pd

from .tseries import tseries_reindex


def read_promice_raw(datafile, reindex=True, **kwargs):
    """Return a DataFrame of pre-processed PROMICE AWS hourly or daily data.

    Reads PROMICE data from raw .txt file and performs these pre-processing
    steps:
    - Construct a datetime index from year, month, day, (hour) columns and
      drop the unneeded year, month, day, etc. columns
    - If `reindex` argument is True, reindex the timeseries to make sure it is
      evenly spaced.
    - Replace -999 flags with NaN
    - Rename IceTemperature8(C) column to IceTemperature10(C) to reflect the
      10m measurement depth.

    **kwargs are keyword arguments to pd.read_csv()
    """

    print('Loading ' + datafile)
    df = pd.read_csv(datafile, sep=' ', **kwargs)

    if 'HourOfDay(UTC)' in df.columns:
        freq = 'H'
    else:
        freq = 'D'

    # Datetime index
    def create_datetime_str(row):
        year, month, day = row['Year'], row['MonthOfYear'], row['DayOfMonth']
        datetime_str = f'{year}-{month:02d}-{day:02d}'
        if 'HourOfDay(UTC)' in row.index:
            hour = row['HourOfDay(UTC)']
            datetime_str = datetime_str + f' {hour:02d}:00'
        return datetime_str

    cols = ['Year', 'MonthOfYear', 'DayOfMonth']
    if freq == 'H':
        cols.append('HourOfDay(UTC)')
    times = df[cols].apply(create_datetime_str, axis=1)
    df['time'] = pd.to_datetime(times)
    df = df.set_index('time').drop(cols, axis=1)
    for col in ['DayOfYear', 'DayOfCentury']:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Make sure timeseries is evenly spaced with no gaps
    if reindex:
        df = tseries_reindex(df, freq=freq)

    # Missing data flagged with -999
    for col in df.columns:
        df.loc[df[col] == -999, col] = np.nan

    # Rename IceTemperature8(C) to reflect the actual depth (10m)
    df = df.rename(columns={'IceTemperature8(C)' : 'IceTemperature10(C)'})

    return df




def read_promice_gps_raw(datafile, stn_name=None, reindex=True,
                         repeats_nan_periods=2, **kwargs):
    """Return a DataFrame of pre-processed PROMICE AWS hourly or daily data
       with some additional steps for GPS measurements.

    Calls read_promice_raw() to read the data, then does some additional
    steps:

    - If stn_name is not None, add a column with stn_name.
    - If reindex is True, reindex the timeseries to make sure it is evenly
      spaced (used in read_promice_raw()).
    - If repeats_nan_periods is not None and latlon is True, replace any
      consecutive periods above repeats_nan_periods of exactly the same latitude
      and longitude with NaN.

    **kwargs are keyword arguments to pd.read_csv()
    """

    df = read_promice_raw(datafile, reindex=reindex, **kwargs)

    # GPS lat, lon
    df['lat'] = df['LatitudeGPS_HDOP<1(degN)']
    df['lon'] = - df['LongitudeGPS_HDOP<1(degW)']

    # Flag repeated latitude + longitude as missing
    if repeats_nan_periods is not None:
        lat, lon = df['lat'], df['lon']
        ind = (lat.diff(periods=1) == 0) & (lon.diff(periods=1) == 0)
        for per in range(2, repeats_nan_periods + 1):
            ind = ind & (lat.diff(periods=per) == 0)
            ind = ind & (lon.diff(periods=per) == 0)
        df.loc[ind, ['lat', 'lon']] = np.nan

    # Add a column with station name
    if stn_name is not None:
        columns = ['Station'] + list(df.columns)
        df['Station'] = stn_name
        df = df[columns]

    return df
