import collections
import numpy as np
import pandas as pd

datadir = '../../data/k_transect/intermed/'
savedir = '../../data/k_transect/processed/'
stations = ['S4', 'S5', 'SHR', 'S6', 'S7', 'S8', 'S9', 'S10']
stations_ab = ['S5', 'S6', 'S9']
datafiles = {station : datadir + station + '.csv' for station in stations}
datafiles_ab = {station : datadir + station + '_ablation.csv' for station in stations_ab}
savefile = savedir + 'k-transect-velocities-hourly.csv'
savefile_daily = savedir + 'k-transect-velocities-daily.csv'
savefile_ab = savedir + 'k-transect-ablation-daily.csv'


# === Ice velocities ===

def read_velocities(filenm):
    print('Reading ' + filenm)
    data_in = pd.read_csv(filenm, index_col=0)
    index = pd.to_datetime(data_in.index)
    u = pd.Series(data_in.iloc[:, 0].values, index=index, name='U')
    return u

# Read hourly velocity data
data = collections.OrderedDict()
for station in stations:
    data[station] = read_velocities(datafiles[station])
data = pd.DataFrame(data)

# Remove S6 wonky days for now
thresh = 250
ind = data['S6'] > thresh
data['S6'][ind] = np.nan

# Resample to daily frequency
data_daily = data.shift(12).resample('24H').mean()

# Save the merged velocity data to csv
print('Saving to ' + savefile)
data.to_csv(savefile)
print('Saving to ' + savefile_daily)
data_daily.to_csv(savefile_daily)


# === Ablation ===

def read_ablation(filenm):
    print('Reading ' + filenm)
    data_in = pd.read_csv(filenm)
    index = pd.to_datetime(data_in[['year', 'month', 'day']])
    ab = pd.Series(data_in.iloc[:, 3].values, index=index, name='ablation')
    return ab

# Read daily ablation data
data_ab = collections.OrderedDict()
for station in stations_ab:
    data_ab[station] = read_ablation(datafiles_ab[station])
data_ab = pd.DataFrame(data_ab)

# Save the merged ablation data to csv
print('Saving to ' + savefile_ab)
data_ab.to_csv(savefile_ab)
