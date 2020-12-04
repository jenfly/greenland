"""
Input:  Raw monthly files from ERA-Interim server with 6-hourly fields of
        analysis fields (multiple variables on lat-lon grid).

Output: For each variable, a file for each year with daily mean fields,
        still on lat-lon grid.
"""
import xarray as xr

import grpy.era as era

datadir = '../../data/era_interim/raw/'
savedir = '../../data/era_interim/latlon/daily/'
filestr = datadir + 'era_interim_an_%d-%02d.nc'
#filestr = datadir + 'era_interim_msl_qflx_%d%02d.nc'

# years = range(1979, 2017)
# months = range(1, 13)
# datafiles = {yr : [filestr % (yr, mon) for mon in months] for yr in years}

years = [2017]
datafiles = {
    2017 : [filestr % (2017, m) for m in range(1, 13)]
}
savestr = savedir + 'era_interim_%s_%d.nc'

def process_file(filenm, var_subset=None):
    print('Loading ' + filenm)
    with xr.open_dataset(filenm) as ds:
        ds = era.cleanup_varids(ds)
        if var_subset is not None:
            ds = ds[var_subset]
        ds_out = era.subdaily_to_daily(ds)
        ds_out.load()
    return ds_out

def process_year(files, var_subset=None, reverse_lat=True, rename_latlon=True):
    data_out = None
    for filenm in files:
        data_in = process_file(filenm, var_subset=var_subset)
        if data_out is None:
            data_out = data_in
        else:
            data_out = xr.concat([data_out, data_in], dim='jday')
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

var_subset = None
for year in years:
    data = process_year(datafiles[year], var_subset=var_subset)
    save_year(data, year, savestr)
