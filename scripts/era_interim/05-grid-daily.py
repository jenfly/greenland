"""
Input:  ERA-Interim daily mean fields on lat-lon grid, and NPLAEA grids to
        interpolate the fields onto.

Output: Data gridded onto polar equal area grid at specified resolution, one
        file per variable.
"""

from mpl_toolkits import basemap
import xarray as xr

import grpy.era as era

#res = '200km'
res = '100km'
years = range(1979, 2018)
varnms = ['viwve', 'viwvn']
#varnms = ['viwve', 'viwvn', 'msl', 'viwvd', 't2m']
#varnms = ['tp']

datadir = '../../data/era_interim/'
savedir = datadir + 'gridded_%s/daily/' % res
filestr = datadir + 'latlon/daily/era_interim_%s_%d.nc'
savestr = savedir + 'era_interim_%s_%d.nc'
mapfile = datadir + 'grids/map.csv'
gridfile = datadir + 'grids/grid_' + res + '.nc'
datafiles = {yr : {nm : filestr % (nm, yr) for nm in varnms} for yr in years}

# Equal area grid for interpolation
with xr.open_dataset(gridfile) as grid:
    grid.load()


def process_year(filenm, varnm, grid):
    print('Loading ' + filenm)
    with xr.open_dataset(filenm) as ds:
        var = ds[varnm]
        var.load()

    print('Interpolating')
    if 'jday' in var.dims:
        tname = 'jday'
        times = var['jday'].values
    else:
        tname = 'time'
        times = var['time'].values
    var_list = []
    for time in times:
        print(time)
        var_list.append(era.interp_var(var.sel(**{tname : time}), grid))

    print('Concatenating')
    var_i = xr.concat(var_list, dim=tname)

    return var_i


for year in years:
    for varnm in varnms:
        filenm = datafiles[year][varnm]
        var_i = process_year(filenm, varnm, grid)
        savefile = savestr % (varnm, year)
        print('Saving to ' + savefile)
        var_i.to_netcdf(savefile)
