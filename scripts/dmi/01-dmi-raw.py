import numpy as np
import os
import pandas as pd
import sys

import grpy as gr

datadir = '../../data/dmi/tr14-08/'
savedir = '../../data/dmi/processed/subdaily/'
savestr = savedir + '%s_subdaily.csv'
statsfile = savedir + 'stats_subdaily.csv'

def get_stats(data):
    stats = data.min().to_frame(name='min')
    stats['max'] = data.max()
    stats = stats.T.reset_index().rename(columns={'index' : 'which'})
    return stats

stns, stats = [], []
for filenm in os.listdir(datadir):
    if filenm.endswith('.txt'):
        stn = filenm.replace('.txt', '')
        stns.append(stn)
        print('Reading ' + datadir + filenm)
        data = gr.read_dmi(datadir + filenm)
        savefile = savestr % stn
        print('Saving to ' + savefile)
        data.to_csv(savefile, index=True)
        stats.append(get_stats(data))

stats = pd.concat(stats, axis=0, ignore_index=True)
cols = ['which', 'station', 'wind dir', 'wind speed (m/s)', 'cloud',
        'slp (hPa)', 'temp drybulb (C)', 'tmax (C)', 'tmin (C)', 'rh (%)',
        'precip (mm)' , 'snow depth (cm)', 'periode']
stats = stats[cols]
print('Saving stats to ' + statsfile)
stats.to_csv(statsfile, index=True)
