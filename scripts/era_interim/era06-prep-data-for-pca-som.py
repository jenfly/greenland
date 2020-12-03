"""
Input:  Monthly and/or daily IVT data gridded onto polar equal area grid.

Output: 2-dimensional array of IVT with all the grid points and x- and y-
        components stacked into columns, and all the time dimensions stacked
        into rows.
"""

import numpy as np
import xarray as xr

import grpy as gr
import grpy.era as era

daily = True
vec = True
#datadir = '../../data/era_interim/gridded_200km/'
datadir = '../../data/era_interim/gridded_100km/'
savedir = '../../data/era_interim/intermed/'
years = np.arange(1979, 2018)

#season = 'JJAS'
#months = [6, 7, 8, 9]

season = 'AMJJAS'
months = [4, 5, 6, 7, 8, 9]

varnms = ['viwve', 'viwvn']
std_vars = False  # Standardize variables

subset = {'x' :  (5.1e6, 6.6e6), 'y' : (2.55e6, 5.25e6)}
savestr = 'ivt_domain04_' + season

# subset = {'x' :  (5.1e6, 6.6e6), 'y' : (2.55e6, 5.25e6)}
# savestr = 'ivt_domain03_' + season

#subset = {'x' :  (4.8e6, 6.9e6), 'y' : (2.3e6, 5.5e6)}
#savestr = 'ivt_domain02_' + season

#subset = {'x' :  (3.6e6, 8e6), 'y' : (1e6, 6e6)}
#savestr = 'ivt_domain01_' + season
#subset = None

yearstr = '_%d-%d.nc' % (min(years), max(years))

if daily:
    filestr = datadir + 'daily/era_interim_%s_%d.nc'
    datafiles = {}
    for yr in years:
        datafiles[yr] = {nm : filestr % (nm, yr) for nm in varnms}
    savefile = savedir + 'era_interim_daily_' + savestr + yearstr
else:
    # Monthly data
    filestr = datadir + 'monthly/era_interim_%s' + yearstr
    datafiles = {nm : filestr % nm for nm in varnms}
    savefile = savedir + 'era_interim_monthly_' + savestr + yearstr


def extract_months(ds, months):
    if 'month' in ds.dims:
        ds_out = gr.subset(ds, {'month' : (months, None)})
    else:
        ds.coords['month'] = ds['time_dt'].dt.month
        ds_out = ds.sel(jday=ds['month'].isin(months))
    return ds_out


def load_data(datafiles, subset=None, months=None):
    """Load data, extract subdomain"""
    ds = xr.Dataset()
    for nm, filenm in datafiles.items():
        print('Loading ' + filenm)
        with xr.open_dataset(filenm) as ds_in:
            ds[nm] = ds_in[nm].load()
    ds = ds.rename({'viwve' : 'ivt_x', 'viwvn' : 'ivt_y'})

    if subset is not None:
        ds = gr.subset(ds, subset)
    if months is not None:
        ds = extract_months(ds, months)

    return ds


# Load data and wrangle time dimensions
if daily:
    for year in years:
        ds_in = load_data(datafiles[year], subset, months)
        ds_in = ds_in.rename({'time_dt' : 'time'}).swap_dims({'jday' : 'time'})

        if year == years[0]:
            data = ds_in
        else:
            data = xr.concat([data, ds_in], dim='time')
else:
    data = load_data(datafiles, subset, season)
    data = data.stack(time=['year', 'month'])
    data = era.expand_multiindex(data, dim='time', vals=data['time_dt'])
    data = data.reset_coords('time_dt', drop=True)

# Stack grid points
data = data.stack(pt=['y', 'x'])
data = data.transpose('time', 'pt')
data = era.expand_multiindex(data, dim='pt')

# IVT vector_amplitude
data['ivt_a'] = era.vector_amplitude(data['ivt_x'], data['ivt_y'])


# Stack variables into a single vector
if vec:
    stack_vars = ['ivt_x', 'ivt_y']
else:
    stack_vars = ['ivt_a']
if 't2m' in varnms:
    stack_vars.append('t2m')
data_vec = era.stack_variables(data, stack_vars=stack_vars, stack_dim='pt',
                               output_name='data_vec', expand_ind=True)

# Standardize variables
def vec_scaling(ivt_x, ivt_y):
    var = xr.concat([ivt_x, ivt_y], dim='new')
    vals = var.values.ravel()
    offset, scale = vals.mean(), vals.std()
    return offset, scale

if std_vars:
    if vec:
        offset, scale = vec_scaling(data['ivt_x'], data['ivt_y'])
        offset_dict = {nm : offset for nm in ['ivt_x', 'ivt_y']}
        scale_dict = {nm : scale for nm in ['ivt_x', 'ivt_y']}
    else:
        offset_dict, scale_dict = {}, {}
    for nm in ['ivt_a', 't2m']:
        if nm in data.data_vars:
            vals = data[nm].values.ravel()
            offset_dict[nm] = vals.mean()
            scale_dict[nm] = vals.std()
    offset_vec = [offset_dict[nm] for nm in data_vec['variable'].values]
    scale_vec = [scale_dict[nm] for nm in data_vec['variable'].values]
    data_vec.coords['offset'] = xr.DataArray(offset_vec, dims=['var'],
                                             coords=data_vec['variable'].coords)
    data_vec.coords['scale'] = xr.DataArray(scale_vec, dims=['var'],
                                            coords=data_vec['variable'].coords)
    data_vec = (data_vec - data_vec['offset']) / data_vec['scale']
    data_vec.name = 'data_vec'

print('Saving to ' + savefile)
data_vec.to_netcdf(savefile)
