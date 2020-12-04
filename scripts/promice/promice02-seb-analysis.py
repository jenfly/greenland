"""
Run SEB analysis on pre-processed hourly PROMICE AWS data.

For each PROMICE station:
- Read the pre-processed .csv file
- Extract columns relevant to SEB analysis
- Interpolate missing data when the gap is sufficiently small
- Calculate net energy fluxes (SW_NET, LW_NET, SEB_NET)
- Calculate melt, sublimation, and ablation
- Calculate solar altitude angle and clear-sky SW radiation
- Save the processed data to a .csv file
"""

import os
import pandas as pd
import pysolar

import grpy as gr

# Data files and stations
# -----------------------
datadir = '../../data/promice_aws/pre-processed/hourly/'
savedir = '../../data/promice_aws/processed/hourly/'
stations_file = '../../data/promice_aws/info/promice_coords.csv'
seb_info_file = f'{savedir}/SEB_info.txt'

stns = [nm.replace('_hourly.csv', '') for nm in os.listdir(datadir) if nm.endswith('_hourly.csv')]
input_files = [f'{datadir}/{stn}_hourly.csv' for stn in stns]
save_files = [f'{savedir}/{stn}_SEB_hourly.csv' for stn in stns]

# Filling missing data
# --------------------
# -- Maximum data gap to fill (# hours) - any bigger gap will be kept as NaNs
interp_lim = 12

# -- Interpolation method
interp_method = 'cubic'


# Utility functions
# -----------------
def calc_solar_altitude(row, lat=None, lon=None):
    timestamp = row.name
    alt = pysolar.solar.get_altitude_fast(lat, lon, timestamp) 
    return alt

def calc_clearsky_radiation(row, alt_thresh=2):  
    timestamp = row.name.tz_localize('utc')
    alt = row['solar_altitude']
    if pd.isnull(alt):
        sw = np.nan
    elif alt < alt_thresh:
        sw = 0
    else:
        sw = pysolar.radiation.get_radiation_direct(timestamp, alt)
    return sw


# Metadata for each station (lat, lon, etc.)
# ------------------------------------------
station_info = pd.read_csv(stations_file, index_col=0)

# Columns to extract and shortened labels to use
# ----------------------------------------------
cols_dict = {'SurfaceTemperature(C)' : 'Ts',
             'AirTemperature(C)' : 'Ta', 
             'ShortwaveRadiationDown_Cor(W/m2)' : 'SW_DN',
             'ShortwaveRadiationUp_Cor(W/m2)' : 'SW_UP',
             'LongwaveRadiationDown(W/m2)' : 'LW_DN',
             'LongwaveRadiationUp(W/m2)' : 'LW_UP',
             'SensibleHeatFlux(W/m2)' : 'SHF',
             'LatentHeatFlux(W/m2)' : 'LHF'}
columns = list(cols_dict.keys())

# Process data and analyze SEB for each station
# ---------------------------------------------
for stn, input_file, save_file in zip(stns, input_files, save_files):
    # Read pre-processed hourly AWS data
    print(f'Reading {input_file}')
    seb = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    # Extract columns of interest and rename with shorter labels  
    seb = seb[columns].rename(columns=cols_dict)
    
    # Fill missings in the raw data
    print('Filling missing data')
    for name in seb.columns:
        print(name)
        output = gr.tseries_fill(seb[name], name=name, freq='H', interp_limit=interp_lim,
                                 interp_method=interp_method)
        df = output['data']
        seb[name] = df[name + '_filled']
        
    
    # Calculate SW_NET, LW_NET, SEB_NET
    print('Calculating net energy fluxes')
    # --- SW and LW components
    for nm in ['SW', 'LW']:
        seb[f'{nm}_NET'] = seb[f'{nm}_DN'] - seb[f'{nm}_UP']

    # --- Net surface energy balance
    cols_sum = ['SW_NET', 'LW_NET', 'SHF', 'LHF']
    seb['SEB_NET'] = seb[cols_sum].sum(axis=1, skipna=False)

    
    # Energy balance model and ablation components
    print('Calculating energy balance model')
    ablation_components = gr.ablation_model(seb)
    seb = seb.join(ablation_components)
    
    # Get metadata for this station
    lat, lon, elev = station_info.loc[stn, ['lat', 'lon', 'elev']]
    
    # Calculate solar altitude
    print('Calculating solar altitude')
    seb['solar_altitude'] = seb.apply(calc_solar_altitude, lat=lat, lon=lon, axis=1)

    # Calculate clear sky SW_DN
    print('Calculating clear sky SW_DN')
    seb['clearsky_SW_DN'] = seb.apply(calc_clearsky_radiation, axis=1)
    
    # Save to .csv file
    print(f'Saving to {save_file}')
    seb.to_csv(save_file)

# Save metadata about the SEB output data
# ---------------------------------------
with open(seb_info_file, 'w') as f:
    f.write('SEB analysis of PROMICE AWS hourly data\n')
    f.write(gr.disptime() + '\n')
    f.write(f'Interpolation method for missing data: {interp_method}\n')
    f.write(f'Maximum data gap filled with interpolation: {interp_lim} hours\n\n')
    f.write(f'Units\n------\n')
    f.write('Energy balance components: W/m2\n')
    f.write('Surface and air temperatures: C\n')
    f.write('Melt, sublimation, and ablation: cm w.e.')