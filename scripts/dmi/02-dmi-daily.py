import numpy as np
import os
import pandas as pd
import sys

import grpy as gr

datadir = '../data/dmi/processed/subdaily/'
savedir = '../data/dmi/processed/daily/'


for filenm in os.listdir(datadir):
    if filenm.endswith('.csv') and not filenm.startswith('stats'):
        datafile = datadir + filenm
        print('Loading ' + datafile)
        data = pd.read_csv(datafile, index_col=0, parse_dates=True)
        if 'periode' in data.columns:
            manual_precip_stn = True
        else:
            manual_precip_stn = False
        df_daily = gr.dmi_daily(data, manual_precip_stn=manual_precip_stn)
        savefile = savedir + filenm.replace('subdaily', 'daily')
        print('Saving to ' + savefile)
        df_daily.to_csv(savefile, index=True)
