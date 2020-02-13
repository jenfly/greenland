import numpy as np
import pandas as pd

from .utils import season_months


def tseries_reindex(data, freq='D', start=None, end=None):
    """Return a timeseries with evenly spaced datetime index.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data with a datetime index
    freq : str
        Frequency of datetime index (e.g. 'D' for daily)
    start, end : str
        Datetime formatted strings for the start and end of the desired date
        range (e.g. start='2017-01-01', end='2017-12-31'). If omitted, the
        datetime index will be constructed from the minimum to the maximum
        datetimes in the input data index.

    Returns
    -------
    data_out : pd.Series or pd.DataFrame
        Data reindexed to an evenly spaced datetime index.
    """
    if start is None:
        tmin = data.index.min()
    else:
        tmin = pd.Timestamp(start)
    if end is None:
        tmax = data.index.max()
    else:
        tmax = pd.Timestamp(end)

    ind = pd.date_range(tmin, tmax, freq=freq)
    return data.reindex(ind, copy=True)


def check_timestamps(data, freq='D', raise_error=False):
    """Return True if index is evenly spaced with no missing timestamps.

    Input 'data' is either pd.Series or pd.DataFrame, with a datetime index
    at frequency specified by input 'freq'.
    """
    data2 = tseries_reindex(data, freq=freq)
    flag = data2.equals(data)
    if flag:
        print('Timestamps ok!')
    else:
        if raise_error:
            raise ValueError('Timestamps unevenly spaced or missing')
        else:
            print('Warning: timestamps unevenly spaced or missing')
    return flag


def repeats_to_nan(data, periods=1):
    """Return timeseries data with repeated values replaced with NaN.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data.
    periods : int
        Number of consecutive periods of repeated values to flag as NaN.
        For example, with daily timeseries data and periods=2, then if a value
        is repeated exactly for 3 days in a row, the data is kept on the first
        day and is replaced with NaN for the next 2 days.

    Returns:
    --------
    data_out : pd.Series or pd.DataFrame
        A copy of the input data with NaNs replacing values flagged as missing.
    """
    if isinstance(data, pd.Series):
        series_flag = True
        data_out = data.to_frame().copy()
    else:
        data_out = data.copy()
        series_flag = False

    for col in data_out.columns:
        ind = data_out[col].diff(periods=1) == 0
        for per in range(2, periods + 1):
            ind = ind & (data_out[col].diff(periods=per) == 0)
        data_out.loc[ind, col] = np.nan
    if series_flag:
        data_out = data_out.iloc[:, 0]
    return data_out


def tseries_resample(data, freq='D', aggregation='mean', hours=None):
    """Return timeseries resampled to specified resolution.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Input data with datetime index.
    freq : string
        Time frequency for resampling. 'D' for daily, 'M' for monthly.
    aggregation : {'mean', 'sum', 'min', 'max'}, optional
        How to aggregate the resampled series.
    hours : list of ints, optional
        Which hours to include (e.g. [6, 18] for precip accumulations at
        6:00 and 18:00).  If omitted then all hours are included.

    Output
    ------
    data_out : pd.Series or pd.DataFrame
        The timeseries at specified resolution.
    """

    if hours is not None:
        data = data[data.index.hour.isin(hours)]
    if aggregation.lower() == 'mean':
        data_out = data.resample(freq).mean()
    elif aggregation.lower() == 'sum':
        data_out = data.resample(freq).sum()
    elif aggregation.lower() == 'min':
        data_out = data.resample(freq).min()
    elif aggregation.lower() == 'max':
        data_out = data.resample(freq).max()
    else:
        raise ValueError('Invalid aggregation ' + aggregation)
    return data_out


def tseries_completeness(tseries, freq='H', start=None, end=None, verbose=True):
    """Return a dict of timeseries completeness stats.

    Parameters
    ----------
    tseries : pd.Series with a DatetimeIndex
        Timeseries of a single variable (not a dataframe)
    freq : {'H', 'D'}, optional
        Frequency of the timeseries (hourly or daily)
    start, end : str
        Datetime formatted strings for the start and end of the desired date
        range (e.g. start='2017-01-01', end='2017-12-31'). If omitted, data
        completeness will be assessed over the range of datetimes in the
        input data index.
    verbose : bool, optional
        If True, print the completeness stats.

    Returns
    -------
    output : dict
        A dict with completeness stats for the timeseries and a series with the
        lengths of each data gap.
    """

    if freq == 'H':
        timedelta_convert = np.timedelta64(1, 'h')
    elif freq == 'D':
        timedelta_convert = np.timedelta64(1, 'D')
    else:
        raise ValueError(f'Invalid freq {freq}. Must be either H or D')

    # Full timeseries with evenly spaced timestamps and NaNs for missings
    ts_full = tseries_reindex(tseries, freq=freq, start=start, end=end)

    # Non-missing subset of timeseries (unevenly spaced timestamps with no NaNs)
    ts_sub = ts_full[ts_full.notnull()]

    # Length of timestamp spacing (in hours)
    ts_spacing = ts_sub.index.to_series().diff() / timedelta_convert

    # Check for missings at beginning and end of timeseries
    if len(ts_sub) > 0:
        # Beginning of timeseries
        t0 = ts_full.index[0]
        t_first = ts_spacing.index[0]
        if t_first != t0:
            ts_spacing.loc[t_first] = (t_first - t0) / timedelta_convert

        # End of timeseries
        t1 = ts_full.index[-1]
        t_last = ts_spacing.index[-1]
        if t_last != t1:
            ts_spacing.loc[t1] = (t1 - t_last) / timedelta_convert

    # Data gaps
    gaps = ts_spacing[ts_spacing > 1] - 1

    # Consolidate info into a dict
    nhrs_total, nhrs_data = len(ts_full), len(ts_sub)
    data_completeness = len(ts_sub) / len(ts_full)
    max_gap = gaps.max()

    stats = pd.Series({'nhrs_total' : nhrs_total,
                       'nhrs_data' : nhrs_data,
                       'data_completeness' : data_completeness,
                       'max_gap' : max_gap})
    output = {'stats' : stats, 'gaps' : gaps}

    if verbose:
        print('Data completeness stats')
        print(f'Total hours: {nhrs_total}\nData hours: {nhrs_data}')
        print(f'Data completeness: {data_completeness:.2%}')
        print(f'Maximum data gap (hours): {max_gap}')

    return output


def tseries_completeness_yrly(data, columns=None, years=None):
    """Return timeseries completeness stats for each year and variable

    Parameters
    ----------
    data : pd.DataFrame
        Input data with a datetime index.
    columns : list of str, optional
        List of columns to include. If omitted, all columns are included.
    years : list of int, optional
        List of years to include. If omitted, all years are included.

    Returns
    -------
    stats_df : pd.DataFrame
        Timeseries completeness and maximum gap size for each variable in
        each year.
    """
    stats_df = []
    if years is None:
        years = range(min(data.index.year), max(data.index.year) + 1)
    if columns is None:
        columns = data.columns
    for year in years:
        start, end = f'{year}-01-01 0:00', f'{year}-12-31 23:59'
        for col in columns:
            tseries = data.loc[f'{year}', col]
            output = tseries_completeness(tseries, start=start, end=end, verbose=False)
            df_in = output['stats']
            df_in['variable'] = col
            df_in = df_in.to_frame(name=year).T
            stats_df.append(df_in)
    stats_df = pd.concat(stats_df)
    return stats_df


def tseries_fill(tseries, name='timeseries', freq='H', start=None, end=None,
                 interp_method='cubic', interp_limit=12, verbose=True):
    """Return a dict with timeseries completeness stats and missings filled.

    Parameters
    ----------
    tseries : pd.Series with a DatetimeIndex
        Timeseries of a single variable (not a dataframe)
    name : str, optional
        Name of timeseries variable (used in columns of output dataframe)
    freq : {'H', 'D'}, optional
        Frequency of the timeseries (hourly or daily)
    start, end : str
        Datetime formatted strings for the start and end of the desired date
        range (e.g. start='2017-01-01 0:00', end='2017-12-31 23:00'). If
        omitted, data completeness will be assessed over the range of datetimes
        in the input data index.
    interp_method : {'time', 'cubic', etc.}, optional
        Interpolation method to fill missings. See documentation of
        pd.Series.interpolate for more options
    interp_limit : int or None, optional
        Maximum number of consecutive missings to fill with interpolation.
    verbose : bool, optional
        If True, print some of the completeness stats.

    Returns
    -------
    output : dict
        A dict with completeness stats for the timeseries, a series with the
        lengths of each data gap, and a dataframe with the original timeseries
        along missings flags and the filled timeseries.
    """

    # Data completeness stats
    output = tseries_completeness(tseries, freq=freq, start=start, end=end,
                                  verbose=verbose)

    # Full timeseries with evenly spaced timestamps and NaNs for missings
    ts_full = tseries_reindex(tseries, freq=freq, start=start, end=end)

    # Dataframe with original timseries, missings flags, and timeseries filled
    # with interpolation
    data = ts_full.to_frame(name=f'{name}_orig')
    data[f'{name}_missing'] = ts_full.isnull()
    data[f'{name}_filled'] = ts_full.interpolate(method=interp_method,
                                                 limit=interp_limit)

    # Since pandas interpolate method does some wonky stuff, manually mask out
    # any gaps that are too big
    def flag_gaps(ts, max_hrs):
        """Return Boolean series of gaps greater than or equal than max_hrs"""
        back_sum = (ts.isnull()).rolling(max_hrs).sum()
        gaps = (back_sum == max_hrs)
        for n in range(1, max_hrs):
            forward_sum = (ts.shift(periods=-n, freq='H').isnull()).rolling(max_hrs).sum()
            gaps = gaps | (forward_sum == max_hrs)
        return gaps

    gaps = flag_gaps(ts_full, max_hrs=interp_limit + 1)
    data.loc[gaps, f'{name}_filled'] = np.nan

    # Consolidate dataframe and other parameters into output dict
    more_output = {'interp_method' : interp_method,
                   'interp_limit' : interp_limit,
                   'data' : data
                   }
    output.update(more_output)

    max_gap = output['stats'].loc['max_gap']
    if verbose and max_gap > interp_limit:
        msg = (f'*** Warning: Biggest gap ({max_gap}) exceeds interp_limit '
               f'({interp_limit}) -- some missings have not been filled. ***')

        print(msg)

    return output


def season_subset(data, season):
    """Return subset of pandas DataFrame or Series for selected season"""
    months = season_months(season)
    ind = [m in months for m in data.index.month]
    data_out = data[ind]
    return data_out
