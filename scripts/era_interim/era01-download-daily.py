"""
Download raw data from ECMWF (6-hourly analysis fields and 3-hourly forecast accumulations, 
on latitude-longitude grid)
"""

import calendar
from ecmwfapi import ECMWFDataServer
import numpy as np
import os

import grpy.era as era

datadir = '../../data/era_interim/raw/'
paramfile = '../../data/era_interim/era_interim_variables.csv'

var_list = ['msl', 'viwve', 'viwvn', 'viwvd','tcwv', 't2m', 'u10', 'v10']
data_type = 'an'
filestr = 'era_interim_an'

# var_list, data_type = ['sf', 'tp'], 'fc'
# filestr = 'era_interim_sf_tp_%4d%02d.nc'
#var_list, data_type = ['tcwv', 't2m', 'u10', 'v10'], 'an'
#filestr = 'era_interim_an_extra_%4d%02d.nc'
# var_list, data_type = ['msl', 'viwve', 'viwvn', 'viwvd'], 'an'
# filestr = 'era_interim_msl_qflx_%4d%02d.nc'

lon1, lon2 = -180, 180
lat1, lat2 = 30, 90
years = [2017, 2018]
#years = np.arange(1979, 2017)
months = np.arange(1, 13)

for year in years:
    for mon in months:
        target = datadir + filestr + f'_{year}-{mon:02d}.nc'
        date1 = f'{year}-{mon:02d}-01'
        ndays = calendar.monthrange(year, mon)[1]
        date2 = f'{year}-{mon:02d}-{ndays:02d}'
        params_all = era.all_parameters(paramfile)
        opts = era.get_opts(var_list, date1, date2, target=target, lat1=lat1,
                            lat2=lat2, lon1=lon1, lon2=lon2,
                            params_all=params_all, data_type=data_type)
        print(target)
        server = ECMWFDataServer()
        server.retrieve(opts)
