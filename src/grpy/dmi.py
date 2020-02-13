import numpy as np
import pandas as pd

from .tseries import tseries_resample


def read_dmi(datafile, check_ranges=True, verbose=False):
    """Return dataframe of DMI data from file, in standardized units."""

    columns = {'stat_no' : 'station',
               'dd' : 'wind dir',
               'ff' : 'wind speed (m/s)',
               'n' : 'cloud',
               'pppp' : 'slp (hPa)',
               'ttt' : 'temp drybulb (C)',
               'txtxtx' : 'tmax (C)',
               'tntntn' : 'tmin (C)',
               'rh' : 'rh (%)',
               'rrr6' : 'precip (mm)',
               'sss' : 'snow depth (cm)'}

    # Some files have wonky wind directions above 400, 500, 600, etc.
    # For now, just leave them as is
    valid_ranges = {'wind dir' : (0, 900),
                    'wind speed (m/s)' : (0, 200),
                    'cloud' : (0, 8),
                    'slp (hPa)' : (700, 1200),
                    'temp drybulb (C)' : (-100, 100),
                    'tmax (C)' : (-100, 100),
                    'tmin (C)' : (-100, 100),
                    'rh (%)' : (0, 110),
                    'precip (mm)' : (0, 1000),
                    'snow depth (cm)' : (0, 1000)}

    data = pd.read_csv(datafile, sep='\t', skipinitialspace=True)
    data['time'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
    data = data.set_index('time').drop(['year', 'month', 'day', 'hour'], axis=1)

    # Special case for a few stations that are precip only
    if 'periode' in data.columns:
        data = data.rename(columns={'stat_no' : 'station',
                                    'precip' : 'precip (mm)'})
        data['precip (mm)'] = 0.1 * data['precip (mm)']
        # Negative precip means more than 0 mm but less than 0.1 mm - fill with 0.01
        data.loc[data['precip (mm)'] < 0, 'precip (mm)'] = 0.01
        return data

    data = data.rename(columns=columns)

    # Variable wind directions - flag as missing
    data.loc[data['wind dir'] >= 900, 'wind dir'] = np.nan

    # Any data value of -9999 flag as missing
    for col in columns.values():
        data.loc[data[col] == -9999, col] = np.nan

    # A few dry bulb temperatures are flagged missing with 9999
    data.loc[data['temp drybulb (C)'] == 9999, 'temp drybulb (C)'] = np.nan

    # Cloud cover 9 means obscured sky due to fog or heavy snow - no data
    data.loc[data['cloud'] == 9, 'cloud'] = np.nan

    # Convert from tenths to listed units
    for nm in ['wind speed (m/s)', 'slp (hPa)', 'temp drybulb (C)', 'tmax (C)',
               'tmin (C)', 'precip (mm)']:
        data[nm] = 0.1 * data[nm]

    # Negative precip means more than 0 mm but less than 0.1 mm - fill with 0.01
    data.loc[data['precip (mm)'] < 0, 'precip (mm)'] = 0.01

    # Snow depth 997 means less than 0.5 cm - fill with 0.1
    # Snow depth 998 means snow cover not continuous - flag as missing
    # Flag any negative snow depths as missing
    data.loc[data['snow depth (cm)'] == 997, 'snow depth (cm)'] = 0.1
    data.loc[data['snow depth (cm)'] == 998, 'snow depth (cm)'] = np.nan
    data.loc[data['snow depth (cm)'] < 0, 'snow depth (cm)'] = np.nan

    # Check that each variable is within a sensible range
    if check_ranges:
        msg = '%s data range (%.0f-%.0f) outside expected range (%.0f-%.0f)'
        for nm, (val_min, val_max) in valid_ranges.items():
            data_min, data_max = data[nm].min(), data[nm].max()
            if (data_min < val_min) or (data_max > val_max):
                raise ValueError(msg % (nm, data_min, data_max, val_min, val_max))

    # Display data ranges
    if verbose:
        stats = data.min().to_frame(name='min')
        stats['max'] = data.max()
        print(stats)

    return data


def dmi_daily(data, exclude_winds=True, manual_precip_stn=False):
    """Return DMI data resampled to daily frequency."""
    index = pd.date_range(data.index.min().date(), data.index.max().date(),
                          freq='D')
    df_daily = pd.DataFrame({'station' : data.loc[data.index[0], 'station']},
                            index=index)

    # Special case for manual precip station
    if manual_precip_stn:
        df_daily['precip (mm)'] = tseries_resample(data['precip (mm)'],
                                                   aggregation='sum',
                                                   hours=None)
        df_daily['periode'] = tseries_resample(data['periode'],
                                               aggregation='sum')
        df_daily = df_daily.fillna(0.0)
        return df_daily

    nms = ['cloud', 'slp (hPa)', 'temp drybulb (C)', 'rh (%)',
          'snow depth (cm)']
    if not exclude_winds:
        nms = ['wind dir', 'wind speed (m/s)'] + nms
    for nm in nms:
        df_daily[nm] = tseries_resample(data[nm], aggregation='mean')
    df_daily['tmax (C)'] = tseries_resample(data['tmax (C)'], aggregation='max')
    df_daily['tmin (C)'] = tseries_resample(data['tmin (C)'], aggregation='min')
    df_daily['precip (mm)'] = tseries_resample(data['precip (mm)'],
                                               aggregation='sum', hours=[6, 18])
    return df_daily


def dmi_yearly(series, buffer_days=5, max_missing=15, verbose=False):
    """Return subset of daily series including only full years.

    A full year defined here has no more than buffer_days missing at the
    beginning or end, and no more than max_missing days missing in total.
    """
    years = list(set(series.index.year))
    series_out = []
    for year in years:
        series_yr = series['%d' % year]
        times = series_yr.index[series_yr.notnull()]
        ndays = len(pd.date_range('%d-01-01' % year, '%d-12-31' % year))
        nmissing = ndays - len(times)
        if verbose:
            print('%d: %d missing' % (year, nmissing))
        if nmissing > max_missing:
            continue
        t0, t1 = min(times), max(times)
        if t0.dayofyear <= buffer_days and t1.dayofyear >= 365 - buffer_days:
            series_out.append(series_yr)
    if len(series_out) == 0:
        series_out = None
    else:
        series_out = pd.concat(series_out)
    return series_out


def dmi_coords(stn_id, coords_lookup):
    coords = coords_lookup.loc[stn_id]
    coords = coords.to_dict()
    coords['station id'] = stn_id
    return coords


def dmi_monthly(df_daily, name='precip (mm)', max_missing=30, verbose=False):
    series = dmi_yearly(df_daily[name], max_missing=max_missing,
                           verbose=verbose)
    if series is None:
        raise ValueError('No full years found in input data')
    if name.lower().find('precip') >= 0:
        aggregation, name_out = 'sum', name
    else:
        aggregation, name_out = 'mean', 'mean ' + name
    df = tseries_resample(series, freq='M', aggregation=aggregation)
    df = df.to_frame(name=name_out)

    # Overall max/min for temperatures
    if name.lower().find('tmax') >= 0:
        df['max ' + name] = tseries_resample(series, freq='M', aggregation='max')
    elif name.lower().find('tmin') >= 0:
        df['min ' + name] = tseries_resample(series, freq='M', aggregation='min')
    return df


def combine_dmi_monthly(df_daily, coords_lookup,
                        columns=['precip (mm)', 'tmax (C)', 'tmin (C)'],
                        max_missing=30, verbose=False):
    stn_id = df_daily['station'][0]
    coords = dmi_coords(stn_id, coords_lookup)
    df_list = []
    for col in columns:
        try:
            df_in = dmi_monthly(df_daily, name=col, max_missing=max_missing,
                                verbose=verbose)
            df_list.append(df_in)
        except KeyError:
            print('Warning: Column %s not found in data, omitting' % col)
        except ValueError:
            print('Warning: No full years for column %s, omitting' % col)
    if len(df_list) > 1:
        df = pd.concat(df_list, axis=1)
    elif len(df_list) == 0:
        raise ValueError('Insufficient data for monthly aggregates')
    else:
        df = df_list[0]
    cols = list(df.columns)
    df['year'] = df.index.year
    df['month'] = df.index.month
    for nm, val in coords.items():
        df[nm] = val
    cols = ['year', 'month', 'station id', 'station name', 'lat', 'lon',
            'elevation (m)'] + cols
    df = df[cols].reset_index(drop=True)
    return df
