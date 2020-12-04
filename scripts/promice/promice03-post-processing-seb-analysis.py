"""
Post-processing for SEB analysis of PROMICE AWS data.

For each PROMICE station:
- Merge hourly data from SEB output .csv files with additional columns from
  pre-processed hourly .csv files.
- Compute additional hourly fields (warming energy, cooling energy, etc.)
- Set any albedos > 1 to NaN
- Set to NaN additional fields which are 0 in the data when they should be NaN
- Compute daily statistics (averages, # hours melting, etc.)
- Save the post-processed hourly and daily data to .csv files.
"""

import os
import numpy as np
import pandas as pd
import calendar

import grpy as gr

# Data files and stations
# -----------------------
datadir = '../../data/promice_aws/pre-processed/hourly/'
sebdir = '../../data/promice_aws/processed/hourly/'
savedir = '../../data/promice_aws/post-processed/'
stns = [nm.replace('_hourly.csv', '') for nm in os.listdir(datadir) if nm.endswith('_hourly.csv')]

# Functions for post-processing
# -----------------------------

def convert_ice_depth(depth_cm_we):
    """Convert ablation/melt/sublimation from cm w.e. to metres of ice."""
    # Densities of ice and water, in kg/m3
    rho_ice = 900
    rho_water = 997

    # Depths in metres of ice
    depth_m_ice = 1e-2 * depth_cm_we * rho_water / rho_ice
    
    return depth_m_ice


def get_times_mask():
    """Return DataFrame of times to manually set the data to NaN"""
    times_mask = [
        ('MIT', '2013-02-23', '2013-06-18', 'both'),
        ('MIT', '2015-01-01', '2015-06-22', 'both'),
        ('MIT', '2016-01-01', '2018-12-31', 'both'),
        ('NUK_K', '2015-04-14', '2015-06-19', 'SEB'),
        ('NUK_K', '2015-03-01', '2015-06-19', 'T'),
        ('NUK_K', '2017-01-01', '2017-05-09', 'SEB'),
        ('NUK_L', '2008-02-07', '2008-02-12', 'both'),
        ('NUK_L', '2011-01-01', '2011-08-25', 'SEB'),
        ('NUK_L', '2011-01-01', '2011-06-08', 'T'),
        ('NUK_L', '2018-01-26', '2018-02-06', 'both'),
        ('NUK_U', '2011-04-01', '2011-08-23', 'SEB'),
        ('NUK_U', '2011-04-01', '2011-06-05', 'T'),
        ('QAS_L', '2007-08-24', '2009-08-31', 'SEB'),   
        ('QAS_U', '2009-04-25', '2009-05-27', 'both'),
        ('QAS_U', '2012-02-13', '2012-06-03', 'both'),
        ('QAS_U', '2014-04-18', '2014-04-25', 'SEB'),
        ('QAS_U', '2015-02-07', '2015-07-09', 'both'),
        ('TAS_A', '2018-01-01', '2018-12-31', 'both'),
        ('TAS_A', '2017-09-27', '2017-12-31', 'both'),
        ('TAS_L', '2011-06-04', '2011-06-06', 'SEB'),
        ('TAS_L', '2012-02-02', '2012-02-03', 'SEB'),
        ('TAS_L', '2012-02-15', '2012-02-16', 'SEB'),
        ('TAS_L', '2012-03-04', '2012-03-27', 'SEB'),
        ('TAS_L', '2015-08-13', '2015-12-31', 'SEB'),
    ]

    times_mask = pd.DataFrame(times_mask, columns=['station', 'start', 'end', 'mask_which'])
    times_mask = times_mask.set_index('station')
    
    return times_mask


def process_hourly(station, datadir, sebdir):
    rawfile = f'{datadir}/{station}_hourly.csv'
    sebfile = f'{sebdir}/{station}_SEB_hourly.csv'

    # SEB fields
    seb_hrly = pd.read_csv(sebfile, index_col=0, parse_dates=True)
    
    # Additional columns from raw data file
    cols_extra = ['Albedo_theta<70d', 'CloudCover', 'DepthPressureTransducer_Cor(m)',
                  'AblationPressureTransducer(mm)']
    for i in list(range(1, 8)) + [10]:
        cols_extra.append(f'IceTemperature{i}(C)')
    data_extra = pd.read_csv(rawfile, index_col=0, parse_dates=True)
    seb_hrly = seb_hrly.join(data_extra[cols_extra])
    
    # Set any albedo > 1 to NaN
    seb_hrly.loc[seb_hrly['Albedo_theta<70d'] > 1, 'Albedo_theta<70d'] = np.nan
    
    # Manually mask out additional spurious values
    times_mask = get_times_mask()
    
    if station in times_mask.index:
        cols_seb = ['SW_DN', 'SW_UP', 'LW_DN', 'LW_UP', 'SHF', 'LHF', 'SEB_NET']
        cols_temp = ['Ts', 'Ta']

        for ind, row in times_mask.loc[[station]].iterrows():
            mask = row['mask_which']
            if mask == 'both':
                cols = cols_seb + cols_temp
            elif mask == 'SEB':
                cols = cols_seb
            elif mask == 'T':
                cols = cols_temp
            else:
                raise ValueError(f'Unknown value for mask_which: {mask}')
            seb_hrly.loc[row['start']:row['end'], cols] = np.nan

        # Apply any additional masking to melt, ablation, etc.
        #melt_missing = ((seb_hrly['Ts'] == 0) & (seb_hrly['SEB_NET'].isnull())) | (seb_hrly['Ts'].isnull())
        melt_missing = seb_hrly['SEB_NET'].isnull() | seb_hrly['Ts'].isnull()
        seb_hrly.loc[melt_missing, 'melt'] = np.nan
        seb_hrly.loc[melt_missing, 'ablation'] = np.nan
        seb_hrly.loc[seb_hrly['LHF'].isnull(), 'sublimation'] = np.nan
 

    # Add cm w.e. units to labels
    cols_dict = {nm : nm + ' (cm w.e.)' for nm in ['melt', 'sublimation', 'ablation']}
    seb_hrly = seb_hrly.rename(columns=cols_dict)
    
    # Add columns with melt, sublimation, ablation in metres ice
    for col in ['melt', 'sublimation', 'ablation']:
        seb_hrly[col + ' (m ice)'] = convert_ice_depth(seb_hrly[col + ' (cm w.e.)'])
    
    # Calculate energy partitioning into melting/warming/cooling/freezing
    melting = (seb_hrly['SEB_NET'] > 0 ) & (seb_hrly['Ts'] == 0)
    seb_hrly['melt_energy'] = seb_hrly['SEB_NET'] * melting

    warming = (seb_hrly['SEB_NET'] > 0) & (seb_hrly['Ts'] < 0)
    seb_hrly['warming_energy'] = seb_hrly['SEB_NET'] * warming

    freezing = (seb_hrly['SEB_NET'] < 0) & (seb_hrly['Ts'] == 0)
    seb_hrly['freezing_energy'] = seb_hrly['SEB_NET'] * freezing

    cooling = (seb_hrly['SEB_NET'] < 0) & (seb_hrly['Ts'] < 0)
    seb_hrly['cooling_energy'] = seb_hrly['SEB_NET'] * cooling

    seb_hrly['freezing/cooling_energy'] = seb_hrly['freezing_energy'] + seb_hrly['cooling_energy']
    
    return seb_hrly


def daily_calcs(seb_hrly, min_hrs=20):
    """Return daily fields, setting to NaN when there are fewer than min_hrs of data in the day."""
    # Mean of each column of hourly data
    seb_daily = seb_hrly.resample('D').mean().drop('solar_altitude', axis=1)
    
    # Melt, ablation, sublimation should be sums rather than means,
    # so update those columns
    cols = ['melt (cm w.e.)', 'ablation (cm w.e.)', 'sublimation (cm w.e.)',
            'melt (m ice)', 'ablation (m ice)', 'sublimation (m ice)']
    seb_daily[cols] = seb_hrly[cols].resample('D').sum()
    
    # Set values to NaN if there aren't enough hourly data points in the day,
    # (except for albedo)
    cols2 = list(seb_hrly.columns)
    cols2.remove('Albedo_theta<70d')
    for col in cols2:
        nhrs_daily = seb_hrly[col].resample('D').count()
        seb_daily.loc[nhrs_daily < min_hrs, col] = np.nan
        
    # Compute SEB components of melt energy (i.e. average only over melt hours)
    cols_seb = ['SW_DN', 'SW_UP', 'LW_DN', 'LW_UP', 'SHF', 'LHF', 'SW_NET', 'LW_NET']
    melting = (seb_hrly['SEB_NET'] > 0 ) & (seb_hrly['Ts'] == 0)
    seb_hrly_melt = seb_hrly[cols_seb].multiply(melting, axis=0)
    seb_hrly_melt.columns = [nm + '_m' for nm in seb_hrly_melt.columns]
    seb_daily_melt = seb_hrly_melt.resample('D').mean()
    for col in seb_daily_melt.columns:
        nhrs_daily = seb_hrly_melt[col].resample('D').count()
        seb_daily_melt.loc[nhrs_daily < min_hrs, col] = np.nan
    seb_daily = seb_daily.join(seb_daily_melt)

    # Min and max Ts, Ta, melt
    ranges = seb_hrly[['Ts', 'Ta', 'melt (cm w.e.)']].resample('D').agg(['min', 'max'])
    ranges.columns = ['_'.join(col) for col in ranges.columns.values]
    for nm in ['Ts', 'Ta', 'melt (cm w.e.)']:
        nhrs_col = seb_hrly[nm].resample('D').count()
        ranges.loc[nhrs_col < min_hrs, nm + '_min'] = np.nan
        ranges.loc[nhrs_col < min_hrs, nm + '_max'] = np.nan
    seb_daily = seb_daily.join(ranges)

    # Number of melt hours
    is_melting = seb_hrly['melt (cm w.e.)'] > 0
    seb_daily['hrs_melt'] = is_melting.resample('D').sum()
    seb_daily.loc[seb_hrly['melt (cm w.e.)'].resample('D').count() < min_hrs, 'hrs_melt'] = np.nan

    # Number of hours with Ts = 0
    sfc_temp_zero = seb_hrly['Ts'] == 0
    seb_daily['hrs_Ts=0'] = sfc_temp_zero.resample('D').sum()
    seb_daily.loc[seb_hrly['Ts'].resample('D').count() < min_hrs, 'hrs_Ts=0'] = np.nan

    # Number of hours with SEB_NET > 0
    pos_seb_net = seb_hrly['SEB_NET'] > 0
    seb_daily['hrs_SEB_NET>0'] = pos_seb_net.resample('D').sum()
    seb_daily.loc[seb_hrly['SEB_NET'].resample('D').count() < min_hrs, 'hrs_SEB_NET>0'] = np.nan
    
    return seb_daily


# Post-process data for each station
# ---------------------------------------------
for stn in stns:
    print(stn)
    seb_hrly = process_hourly(stn, datadir, sebdir)
    savefile1 = f'{savedir}/{stn}_SEB_hourly.csv'
    print('Saving to ' + savefile1)
    seb_hrly.to_csv(savefile1, index_label='Datetime')
    
    seb_daily = daily_calcs(seb_hrly)
    savefile2 = f'{savedir}/{stn}_SEB_daily.csv'
    print('Saving to ' + savefile2)
    seb_daily.to_csv(savefile2, index_label='Datetime')
    

# Data completeness stats for each station
# -------------------------------------------------

# Remove EGP from stations list because there is zero melt in all years
stns.remove('EGP')

datafiles_hrly = {stn: savedir + stn + '_SEB_hourly.csv' for stn in stns}
datafiles_daily = {stn: savedir + stn + '_SEB_daily.csv' for stn in stns}
savefile_hrly = savedir + 'ablation_hourly_completeness_stats.csv'
savefile_daily = savedir + 'ablation_daily_completeness_stats.csv'

# Read post-processed data from each station
data_hrly, data_daily = {}, {}
for stn in stns:
    data_hrly[stn] = pd.read_csv(datafiles_hrly[stn], index_col=0, parse_dates=True)
    data_daily[stn] = pd.read_csv(datafiles_daily[stn], index_col=0, parse_dates=True)

# Compute completeness stats
def get_completeness(ts_in, start, end, freq='H'):
    output = gr.tseries_completeness(ts_in, start=start, end=end, freq=freq, verbose=False)
    return output['stats']['data_completeness']
        
def get_stats(data, stns, freq='H'):
    stats_df = pd.DataFrame()
    for stn in stns:
        df = data[stn]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for year in df.index.year.unique():
            ts_yr = df.loc[f'{year}', 'ablation (cm w.e.)']
            stats = pd.DataFrame({'Station' : stn, 'Year' : year}, index=[0])

            # Full year
            stats['Year-Completeness'] = get_completeness(ts_yr, start=f'{year}-01-01', 
                                                          end=f'{year}-12-31', freq=freq)

            # Main melt season Jun-Aug
            start, end = f'{year}-06-01', f'{year}-08-31'
            stats['JJA-Completeness'] = get_completeness(ts_yr.loc[start:end], start=start, 
                                                         end=end, freq=freq)
            
            # Longer melt season May-Sep
            start, end = f'{year}-05-01', f'{year}-09-30'
            stats['May-Sep-Completeness'] = get_completeness(ts_yr.loc[start:end], start=start, 
                                                             end=end, freq=freq)
            
            # Apr-Sep
            start, end = f'{year}-04-01', f'{year}-09-30'
            stats['Apr-Sep-Completeness'] = get_completeness(ts_yr.loc[start:end], start=start, 
                                                             end=end, freq=freq)
            
            # May-Aug
            start, end = f'{year}-05-01', f'{year}-08-31'
            stats['May-Aug-Completeness'] = get_completeness(ts_yr.loc[start:end], start=start, 
                                                             end=end, freq=freq)

            # Monthly stats
            for m, month in enumerate(months, start=1):
                start = f'{year}-{m:02d}-01'
                ndays = calendar.monthrange(year, m)[1]
                end = f'{year}-{m:02d}-{ndays:02d}'
                try:
                    ts_month = ts_yr.loc[f'{year}-{m:02d}']
                    stats[f'{month}-Completeness'] = get_completeness(ts_month, start=start, 
                                                                      end=end, freq=freq)
                except KeyError:
                    stats[f'{month}-Completeness'] = 0

            # Append to summary dataframe
            stats_df = stats_df.append(stats, ignore_index=True)
    
    return stats_df

stats_hrly = get_stats(data_hrly, stns, freq='H')
stats_daily = get_stats(data_daily, stns, freq='D')

# Save to file
print(f'Saving to {savefile_hrly}')
stats_hrly.to_csv(savefile_hrly, index=False)
print(f'Saving to {savefile_daily}')
stats_hrly.to_csv(savefile_daily, index=False)
