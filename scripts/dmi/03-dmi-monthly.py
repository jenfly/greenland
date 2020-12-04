import numpy as np
import os
import pandas as pd
import sys

import grpy as gr

datadir = '../../data/dmi/processed/daily/'
savefile = '../../data/dmi/processed/dmi_monthly.csv'

coords_lookup = pd.read_csv('../../data/dmi/metadata/dmi_coords.csv', index_col=0)
coords_lookup = coords_lookup[['station name', 'lat', 'lon', 'elevation (m)']]

df_all = []
skipped = []
for filenm in os.listdir(datadir):
    if filenm.endswith('.csv') and not filenm.startswith('stats'):
        datafile = datadir + filenm
        print('Loading ' + datafile)
        df_daily = pd.read_csv(datafile, index_col=0, parse_dates=True)
        try:
            df_in = gr.combine_dmi_monthly(df_daily, coords_lookup, verbose=True)
            df_all.append(df_in)
        except ValueError:
            print('*** Warning: Insufficient data for monthly aggregates, omitting ***')
            skipped.append(filenm)
            continue
df_all = pd.concat(df_all, axis=0, ignore_index=True)
cols = ['year', 'month', 'station id', 'station name', 'lat', 'lon',
        'elevation (m)', 'precip (mm)', 'mean tmax (C)', 'max tmax (C)',
        'mean tmin (C)', 'min tmin (C)']
df_all = df_all[cols]
print('Saving to ' + savefile)
df_all.to_csv(savefile)

print('Skipped files:')
print(skipped)
