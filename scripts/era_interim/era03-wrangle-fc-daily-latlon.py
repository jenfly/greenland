"""
Input:  Raw monthly files from ERA-Interim server with 3-hourly fields of
        forecast accumulations (multiple variables on lat-lon grid).

Output: For each variable, a file for each year with daily total accumulations
        fields, still on lat-lon grid.
"""
import pandas as pd
import xarray as xr

import grpy.era as era

datadir = '../../data/era_interim/raw/'
savedir = '../../data/era_interim/latlon/daily/'
years = range(1979, 2017)
months = range(1, 13)
hr_res = 3 # 3-hourly data
filestr = datadir + 'era_interim_sf_tp_%d%02d.nc'
#filestr = datadir + 'era_interim_msl_qflx_%d%02d.nc'
datafiles = {yr : [filestr % (yr, mon) for mon in months] for yr in years}
savestr = savedir + 'era_interim_%s_%d.nc'
var_subset = None
paramfile = '../params/era_interim_variables.csv'

def process_file(filenm, var_subset=None, hr_res=3, params_df=None):
    print('Loading ' + filenm)
    with xr.open_dataset(filenm) as ds:
        if var_subset is not None:
            ds = ds[var_subset]

        # Compute daily sums
        times = pd.to_datetime(ds['time'].values)
        times = times.shift(-hr_res, freq='H')
        ds['time'] = times
        ds = ds.resample('D', dim='time', how='sum')
        ds.load()
    return ds

def process_year(files, var_subset=None, reverse_lat=True, rename_latlon=True,
                 hr_res=3, params_df=None):
    data_out = None
    for filenm in files:
        data_in = process_file(filenm, var_subset=var_subset, hr_res=hr_res,
                               params_df=params_df)
        if data_out is None:
            data_out = data_in
        else:
            data_out = xr.concat([data_out, data_in], dim='time')
    # Reverse latitudes so that they are ascending
    if reverse_lat:
        lat = data_out['latitude'][::-1]
        data_out = data_out.reindex_like(lat)
    if rename_latlon:
        data_out = data_out.rename({'latitude' : 'lat', 'longitude' : 'lon'})
    return data_out

def save_year(data, year, savestr):
    for nm in data.data_vars:
        savefile = savestr % (nm, year)
        print('Saving to ' + savefile)
        data[nm].to_netcdf(savefile)

params_df = era.all_parameters(paramfile)

for year in years:
    data = process_year(datafiles[year], var_subset=var_subset, hr_res=hr_res,
                        params_df=params_df)
    save_year(data, year, savestr)
