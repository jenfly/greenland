"""
Pre-process hourly PROMICE AWS data from raw (.txt) files.

For each PROMICE station:
- Read the hourly data from the .txt file
- Construct a datetime index from year, month, day, (hour) columns and
  drop the unneeded year, month, day, etc. columns
- Reindex the timeseries to make sure it is evenly spaced.    
- Replace -999 flags with NaN
- Save the pre-processed data to a .csv file
- Create missings flags for SEB components and Ts, and save to .csv file
"""

import os
import pandas as pd

import grpy as gr

# User input parameters
# ---------------------
datadir = '../../data/promice_aws/raw/hourly/'
savedir = '../../data/promice_aws/pre-processed/hourly/'

# Data files and list of stations
# -------------------------------
stns = [nm.replace('_hour.txt', '') for nm in os.listdir(datadir) if nm.endswith('_hour.txt')]

# Pre-process data for each station
# -----------------------------
for stn in stns:
    # Read raw data from text file and pre-process
    input_file = f'{datadir}/{stn}_hour.txt'
    data = gr.read_promice_raw(input_file)
    
    # Save to .csv file
    output_file = f'{savedir}/{stn}_hourly.csv'
    print(f'Saving to {output_file}')
    data.to_csv(output_file, index_label='Datetime')
    
    # Generate missings flags for SEB components and surface temperature
    cols_seb = ['ShortwaveRadiationDown_Cor(W/m2)', 'ShortwaveRadiationUp_Cor(W/m2)',
                'LongwaveRadiationDown(W/m2)', 'LongwaveRadiationUp(W/m2)',
                'SensibleHeatFlux(W/m2)', 'LatentHeatFlux(W/m2)']

    missings = data[cols_seb].isnull()
    missings['SEB_any'] = missings.any(axis=1)
    missings['Ts'] = data['SurfaceTemperature(C)'].isnull()
    missings = missings.astype(int)
    
    # Save missings flags to .csv file
    missings_file = f'{savedir}/{stn}_hourly_SEB_Ts_missings.csv'
    print(f'Saving to {missings_file}')
    missings.to_csv(missings_file, index_label='Datetime')
    
    