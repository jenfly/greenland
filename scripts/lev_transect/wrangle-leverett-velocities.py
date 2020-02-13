"""
Process  ice velocities from .mat files and save to CSV files.
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr

import grpy as gr

datadir = '../../data/lev_transect/raw/velocities/'
savedir = '../../data/lev_transect/processed/'

# Option to save intermediate data for troubleshooting
save_extra = False
savedir_extra = '../../data/lev_transect/intermed/'

# Process either v_6h or v_24h
varnm = 'v_6h'
savefile = savedir + 'lev_' + varnm + '_daily.csv'

# Filenames for input data
stns = ['lev%d' % i for i in range(9)]
filestr = datadir + '%svel_2009_spr2013_UTC-2_20130528.mat'
datafiles = {stn : filestr % stn for stn in stns}


def process_var(vals, name, day0='2008-12-31', verbose=True):
    sec_per_day = 24 * 60 * 60

    df = pd.DataFrame(vals[:, :2], columns=['day', name])
    df = df[df['day'].notnull()]
    df['sec'] = np.round(df['day'] * sec_per_day, decimals=0)

    print('\nComputing timestamps and datetimes')
    t0 = pd.Timestamp(day0)
    t0 = t0.value/1e9
    df['timestamp'] = t0 + df['sec']
    df['timestamp'] = df['timestamp'].astype('int32')
    df['time'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('time')
    df = df[[name, 'timestamp']]
    if verbose:
        print(df.head())
        print(df.tail())

    return df


# Iterate over each station
df_daily = {}
for stn in stns:
    filenm = datafiles[stn]
    print('Loading ' + filenm)
    data_in = sio.loadmat(filenm)

    # Process data - calculate timestamps, datetimes
    df = process_var(data_in[varnm], name=varnm)

    # If saving intermediate data: convert to xarray dataset and save to netcdf
    if save_extra:
        ds = xr.Dataset(df)
        ds = ds.set_coords('timestamp')
        savefile_extra = savedir_extra + stn + '_' + varnm + '.nc'
        print('Saving to ' + savefile_extra)
        ds.to_netcdf(savefile_extra)

    # Daily velocities
    if varnm == 'v_6h':
        df_daily[stn] = df[varnm].resample('D').mean()
    else:
        df_daily[stn] = df[varnm]

# Merge daily velocities from all stations
df_daily = pd.DataFrame(df_daily)

# Check that merged data has evenly spaced timestamps
gr.check_timestamps(df_daily, freq='D', raise_error=True)

print('Saving daily data to ' + savefile)
df_daily.to_csv(savefile)
