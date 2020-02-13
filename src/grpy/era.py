from ecmwfapi import ECMWFDataServer
from mpl_toolkits import basemap
import numpy as np
import pandas as pd
import sys
import xarray as xr


def all_parameters(datafile=None):
    """Return a DataFrame of parameter codes and info for ERA-Interim.

    Input datafile is a csv file with all the codes and info.  If datafile is
    None, then the input file defaults to:
    '../params/era_interim_variables.csv'
    """
    if datafile is None:
        datafile = ('../params/era_interim_variables.csv')
    print('Reading variable IDs and info from ' + datafile)
    df = pd.read_csv(datafile, index_col=0, dtype=str)
    df['code'] = df['code'].astype(str)
    return df


def param_str(var_list, params_all=None):
    """Return a string for keyword `param` to specify ERA-Interim variables."""
    if params_all is None:
        params_all = all_parameters()
    return '/'.join(params_all['code'].loc[var_list])


def date_str(date1, date2):
    """Return a string for keyword `date` for ERA-Interim download.

    Inputs date1, date2 should be strings in the format yyyy-mm-dd
    """
    return '%s/to/%s' % (date1, date2)


def area_str(lat1, lat2, lon1, lon2):
    """Return a string for keyword `area` to download ERA-Interim lat-lon subset"""
    return '%d/%d/%d/%d' % (lat2, lon1, lat1, lon2)


def get_opts(var_list, date1, date2, target='output.nc', lat1=None, lat2=None,
             lon1=None, lon2=None, data_type='an', params_all=None):
    """Return dictionary of options to input to ECMWFDataServer.retrieve()

    Need to test on forecast variables, and add error checking to make sure
    all input variables are of the same data type (analysis or forecast).
    """
    date = date_str(date1, date2)
    param = param_str(var_list, params_all)
    if data_type == 'an':
        step = '0'
        time = '00:00:00/06:00:00/12:00:00/18:00:00'
    elif data_type == 'fc':
        step = '3/6/9/12'
        time = '00:00:00/12:00:00'

    opts = {'class': 'ei',
            'dataset': 'interim',
            'date': date,
            'expver': '1',
            'grid': '0.75/0.75',
            'levtype': 'sfc',
            'param': param,
            'step': step,
            'stream': 'oper',
            'time': time,
            'type': data_type,
            'format' : 'netcdf',
            'target': target
            }
    if lon1 is not None:
        area = area_str(lat1, lat2, lon1, lon2)
        opts['area'] = area

    return opts


def cleanup_varids(ds, params_df=None):
    """Replace numeric IDs in ERA-Interim dataset with descriptive short IDs.
    """
    if params_df is None:
        params_df = all_parameters()
    reverse = {params_df['code'].loc[nm] : nm for nm in params_df.index }
    reverse_sub = {}
    for nm in reverse:
        if 'p' + nm in ds.data_vars:
            reverse_sub['p' + nm] = reverse[nm]
    return ds.rename(reverse_sub)


def time_multiindex(var, time_name='time', year=True, month=True, day=True,
                    hour=False, jday=False, copy=False):
    """Return a DataArray with a multiindex of the time coordinate.

    If copy is False, the new coordinates are also added to the input variable.
    Otherwise, a copy of the input variable is made before the coordinates are
    changed and the input variable is not modified.
    """

    if copy:
        var_out = var.copy()
    else:
        var_out = var

    # Preserve the original time coordinate as another coordinate
    var_out.coords[time_name + '_dt'] = var_out[time_name]
    time = var[time_name].dt
    time_vars = {'year' : time.year, 'month' : time.month,
                 'day' : time.day, 'jday' : time.dayofyear,
                 'hour' : time.hour
                 }
    time_list = []
    pairs = zip(['year', 'month', 'day', 'jday', 'hour'],
                [year, month, day, jday, hour])
    for tdim, incl in pairs:
        if incl:
            var_out.coords[tdim] = time_vars[tdim]
            time_list.append(tdim)

    var_out = var_out.set_index(**{time_name : time_list})
    return var_out


def stack_time(var_in, time_dims=['year', 'month', 'day'],
               time_name='time'):
    """Return DataArray with time dimensions stacked."""

    var_out = var_in.stack(**{time_name : time_dims})
    dims = list(var_out.dims)
    dims.remove(time_name)
    dims = [time_name] + dims
    var_out = var_out.transpose(*dims)
    return var_out


def unstack_time(var_in, time_dims=['year', 'month', 'day'],
                 time_name='time'):
    """Return DataArray with time dimensions unstacked."""

    dims = list(var_in.dims)
    dims.remove(time_name)
    dims = time_dims + dims
    var_out = var_in.unstack(time_name).transpose(*dims)
    return var_out


def expand_multiindex(data, dim, vals=None, copy=False):
    """Expand xarray multiindex so that it can be saved to netcdf.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
    dim :  str, name of multiindex dimension to expand
    vals : np.array, optional array of values for new index. Default is
             np.arange(len(data[dim]))
    copy : bool, optional.  If False, then some extra coords are added to
             the input data.

    Returns
    -------
    data_out : xr.DataArray or xr.Dataset
    """

    if copy:
        data_out = data.copy()
    else:
        data_out = data
    nms = data.indexes[dim].names
    dim2 = dim + '_vals'

    if vals is None:
        vals = np.arange(len(data[dim]))

    data_out[dim2] = xr.DataArray(vals, dims=[dim])

    # Save the component coordinates of the multiindex so they don't
    # get deleted when resetting the index
    save_nms = {nm : 'save_' + nm for nm in nms}
    for nm1, nm2 in save_nms.items():
        data_out.coords[nm2] = data_out[nm1]

    # Reset index
    data_out = data_out.set_index(**{dim: dim2})

    # Cleanup: return component coords to their original names
    rev_nms = {val : nm for nm, val in save_nms.items()}
    data_out = data_out.rename(rev_nms)

    return data_out


def stack_variables(ds, stack_vars=['ivt_x', 'ivt_y'], stack_dim='pt',
                    output_name='ivt_vec', expand_ind=True):
    """Stack variables in xr.Dataset into a single variable"""
    var = ds[stack_vars].to_array().stack(var=['variable', stack_dim])
    if expand_ind:
        var = expand_multiindex(var, dim='var')
        var['variable'] = var['variable'].astype(str)
    var.name = output_name
    return var


def unstack_variables(var, var_dim='var', comp_dims=['pt', 'variable'],
                      cleanup=True):
    """Unstack variables in a single xr.DataArray into a Dataset"""
    if comp_dims is not None:
        var = var.set_index(**{var_dim : comp_dims})
    var_unstk = var.unstack(var_dim)
    
    if cleanup:    
        # Remove extra variable dimension from any coordinates
        variable_coords = var_unstk['variable']
        var0 = variable_coords.values[0]
        var0 = str(var_unstk['variable'][0].values)
        for nm in list(var_unstk.coords.keys()):
            dims = var_unstk[nm].dims
            if len(dims) > 1 and 'variable' in dims:
                var_unstk[nm] = var_unstk[nm].sel(variable=var0).drop('variable')
        ds_out = var_unstk.to_dataset(dim='variable')
    else:
        ds_out = var_unstk.to_dataset(dim='variable')
    return ds_out


def subdaily_to_daily(data, time_name='time', squeeze_years=True):
    """Compute daily means from subdaily data.

    Input data is xr.DataArray or xr.Dataset."""

    years = data[time_name].dt.year
    ny = max(years) - min(years) + 1
    if squeeze_years and ny == 1:
        keep_year, time_dims = False, ['jday', 'hour']
    else:
        keep_year, time_dims = True, ['year', 'jday', 'hour']

    # Unstack time dimensions
    data_out = time_multiindex(data, year=keep_year, month=False, day=False,
                               jday=True, hour=True, time_name=time_name)
    data_out = unstack_time(data_out, time_dims=time_dims, time_name=time_name)

    # Preserve datetime array
    dtname = time_name + '_dt'
    if dtname in data_out:
        time_dt = data_out[dtname].sel(hour=0)
        dims = time_dt.dims
        coords = {nm : time_dt[nm] for nm in dims}
        time_dt = xr.DataArray(time_dt.values, dims=dims, coords=coords)
        data_out[dtname] = time_dt

    # Daily mean
    return data_out.mean(dim='hour', keep_attrs=True)


def interp_var(var_in, grid, xname='x', yname='y', lon_name='lon',
               lat_name='lat', keep_attrs=True):
    """Interpolate lat-lon data onto another grid.

    Input var_in is a DataArray with latitude as the second-last dimension and
    longitude as the last dimension.
    """
    if keep_attrs:
        attrs = var_in.attrs
    else:
        attrs = None

    coords = dict(grid.coords)
    var_coords = dict(var_in.coords)
    for nm in var_coords:
        nm != lon_name and nm != lat_name
        coords[nm] = var_coords[nm]
    lons, lats = var_in[lon_name], var_in[lat_name]
    grid_lons, grid_lats = grid[lon_name], grid[lat_name]
    coords[lon_name], coords[lat_name] = grid_lons, grid_lats
    vals = basemap.interp(var_in.values, lons, lats, grid_lons, grid_lats)
    var_i = xr.DataArray(vals, dims=[yname, xname], coords=coords, attrs=attrs,
                         name=var_in.name)
    return var_i


def vector_amplitude(vec_x, vec_y):
    """Return the amplitude of a vector field"""
    return np.sqrt(np.square(vec_x) + np.square(vec_y))


def add_gridcell_edges(data):
    """Add x and y coordinates for grid cell edges."""
    data_unstk = data.set_index(pt=['y', 'x']).unstack('pt')
    dx = min(abs(np.diff(data['x']))) / 2.0
    dy = max(abs(np.diff(data['y']))) / 2.0

    x = data_unstk['x'].values - dx
    x = np.concatenate([x, [x[-1] + 2*dx]])
    y = data_unstk['y'].values - dy
    y = np.concatenate([y, [y[-1] + 2*dy]])

    data.coords['x_edges'] = xr.DataArray(x, dims=['ix'])
    data.coords['y_edges'] = xr.DataArray(y, dims=['iy'])

    # xleft, xright = x[:-1], x[1:]
    # ybottom, ytop = y[:-1], y[1:]
    # data_unstk['x_left'] = xr.DataArray(xleft, dims=['x'])
    # data_unstk['x_right'] = xr.DataArray(xright, dims=['x'])
    # data_unstk['y_bottom'] = xr.DataArray(ybottom, dims=['y'])
    # data_unstk['y_top'] = xr.DataArray(ytop, dims=['y'])
    #
    # for nm in ['x_left', 'x_right', 'y_bottom', 'y_top']:
    #     data.coords[nm] = data_unstk[nm].stack(pt=['y', 'x'])

    return data
