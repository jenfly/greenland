import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from geopy.distance import vincenty
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import xarray as xr


def geodist(lat1, lon1, lat2, lon2):
    """Return the Vincenty distance in metres from (lat1, lon1) to (lat2, lon2).
    """
    return vincenty((lat1, lon1), (lat2, lon2)).meters

def country_boundary(country_name, resolution='110m', polygon_index=-1):
    shpfilename = shpreader.natural_earth(resolution=resolution, category='cultural',
                                          name='admin_0_countries')

    reader = shpreader.Reader(shpfilename)

    for country in reader.records():
        if country.attributes['NAME_LONG'].lower() == country_name.lower():
            break
    if country.attributes['NAME_LONG'].lower() != country_name.lower():
        raise ValueError('Country ' + country_name + ' not found.')
    polygon = country.geometry[polygon_index]
    country_x, country_y = polygon.exterior.coords.xy
    country_x, country_y = np.array(country_x), np.array(country_y)
    return country_x, country_y


def domain_crs(central_latitude=90, central_longitude=-40):
    """Return Lambert Azimuthal Equal Area crs for Greenland maps."""
    return ccrs.LambertAzimuthalEqualArea(central_latitude=central_latitude,
                                          central_longitude=central_longitude)


def domain_map(res='50m', land_kw={'color' : '0.7'}, gridlines_kw={'color' : '0.8'},
               greenland_zoom=False, xlims=(-2e6, 2e6), ylims=(-3.8e6, 0),
               crs=None, fig=None, nrows=1, ncols=1, subplot=1):
    """Plot Greenland domain map and return ax, crs."""
    if greenland_zoom:
        xlims = -0.9e6, 0.8e6
        ylims = -3.5e6, -0.5e6
    if crs is None:
        crs = domain_crs()
    if fig is None:
        fig = plt.gcf()
    ax = fig.add_subplot(nrows, ncols, subplot, projection=crs)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    cf = cfeature.NaturalEarthFeature('physical', 'land', res)
    ax.add_feature(cf, **land_kw)
    ax.gridlines(crs=ccrs.Geodetic(), **gridlines_kw)
    return ax, crs


def make_box(xmin, xmax, ymin, ymax, xname='x', yname='y', nseg_x=1, nseg_y=1):
    """Return DataFrame of points enclosing box bounded by the inputs.

    Call signature : box = make_box(xmin, xmax, ymin, ymax, **kwargs)

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Boundaries to define the box
    xname, yname : str, optional
        Name of x and y coordinates in output DataFrame
    nseg_x, nseg_y : int, optional
        Number of line segments to construct each side of the box.

    Returns
    -------
    box : pandas.DataFrame
        x, y coordinates of each point to construct the box
    """

    def get_segments(x1, x2, nseg):
        return np.linspace(x1, x2, nseg + 1)

    side_x = get_segments(xmin, xmax, nseg_x)
    side_y = get_segments(ymin, ymax, nseg_y)
    left = list(zip([xmin] * len(side_y), side_y))
    top = list(zip(side_x, [ymax] * len(side_x)))
    right = list(zip([xmax] * len(side_y), side_y[::-1]))
    bottom = list(zip(side_x[::-1], [ymin] * len(side_x)))
    pts = left[:-1] + top[:-1] + right[:-1] + bottom
    x, y = list(zip(*pts))
    box = pd.DataFrame({xname : x, yname : y}, columns=[xname, yname])
    return box


def plot_box(xmin, xmax, ymin, ymax, **kwargs):
    """Plot a box bounded by the min/max x,y provided.

    **kwargs are optional keyword arguments to plt.plot()
    """
    x = [xmin, xmin, xmax, xmax, xmin]
    y = [ymin, ymax, ymax, ymin, ymin]
    plt.plot(x, y, **kwargs)
    return None


def inside_polygon(x, y, poly_x, poly_y):
    """Return indices of points inside the polygon.

    Inputs x, y must be 1-dimensional arrays, so if they are points from
    np.meshgrid(), then apply .flatten() to each first.
    """
    def vecs_to_array(x, y):
        x = x.reshape((len(x), 1))
        y = y.reshape((len(y), 1))
        return np.concatenate([x, y], axis=1)

    if isinstance(x, xr.DataArray) or isinstance(x, pd.Series):
        x, y = x.values, y.values
    if isinstance(poly_x, xr.DataArray) or isinstance(poly_x, pd.Series):
        poly_x, poly_y = poly_x.values, poly_y.values

    polygon = vecs_to_array(poly_x, poly_y)
    points = vecs_to_array(x, y)
    path = Path(polygon)
    inside = path.contains_points(points)
    return inside



###### Old functions for using basemap


def convert_coords(xin, yin, m, big=1e20, inverse=False):
    """Return x, y coords in Basemap m projection, accounting for NaNs.

    Call signature : xout, yout = convert_coords(xin, yin, m, big=big)

    Parameters
    -----------
    xin, yin : np.array, pd.Series, or xarray.DataArray
        Arrays of coordinates of each point (can be vectors or
        matrices) in lon-lat or map coordinates
    m : Basemap object
        Map projection to use
    big : float, optional
        Value to flag NaNs output from m(xin, yin).  Any point with
        absolute value greater than big is replaced with np.nan
    inverse : bool, optional
        If False, convert from lon-lat to map coordinates.  If True, convert
        from map coordinates to lon-lat.

    Returns
    -------
    xout, yout : np.array
        Arrays of x, y coordinates of each point converted from lon-lat to
        map coordinates (inverse=False) or from map coordinates to lon-lat
        (inverse=True)
    """

    if isinstance(xin, pd.Series) or isinstance(xin, xr.DataArray):
        xout, yout = m(xin.values, yin.values, inverse=inverse)
    else:
        xout, yout = m(xin, yin, inverse=inverse)
    xout[abs(xout) > big] = np.nan
    yout[abs(yout) > big] = np.nan
    return xout, yout


def plot_parallels_meridians(m, parallels=[], meridians=[], latmin=0,
                            latmax=90, lonmin=-180, lonmax=180,
                            nx_parallels=1000, ny_meridians=1000, **kwargs):
    """Plot parallels and meridians on a map.

    Parameters
    ----------
    m : Basemap object
        Map projection to convert lat-lon into map coordinates
    parallels, meridians : list or np.array, optional
        Latitudes of parallels to plot and longitudes of meridians to
        plot.  To omit either one, set to an empty list [].
    latmin, latmax, lonmin, lonmax : float, optional
        Extent of meridians (latmin, latmax) and parallels (lonmin, lonmax)
        to draw.
    nx_parallels, ny_meridians : int, optional
        Number of points to create line segments with np.linspace().
    **kwargs : keyword arguments, optional
        Keyword arguments to plt.plot()
    """
    parallels_lons = np.linspace(lonmin, lonmax, nx_parallels)
    meridians_lats = np.linspace(latmin, latmax, ny_meridians)

    for lat in parallels:
        xs, ys = m(parallels_lons, [lat] * len(parallels_lons))
        plt.plot(xs, ys, **kwargs)
    for lon in meridians:
        xs, ys = m([lon] * len(meridians_lats), meridians_lats)
        plt.plot(xs, ys, **kwargs)
    return None


def format_ticks(xfmt='%.1e', yfmt='%.1e'):
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter(xfmt))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter(yfmt))
    return None


def get_map_info(m):
    """Return dict of parameters describing the Basemap m"""
    nms = ['projection', 'ellipsoid', 'boundinglat', 'projparams', 'round']
    vals = [m.projection, m.ellipsoid, m.boundinglat, m.projparams, m.round]
    map_info = {nm : val for nm, val in zip(nms, vals)}
    return map_info


def map_from_csv(csvfile, return_dict=True):
    """Return Basemap object or dict of Basemap parameters from .csv file"""
    df = pd.read_csv(csvfile)

    # Replace NaNs with None
    df = df.where((pd.notnull(df)), None)
    map_opts = (df.iloc[0]).to_dict()

    if not return_dict:
        m = Basemap(**map_opts)
        output = m
    else:
        output = map_opts

    return output


def equal_area_grid(xmin, xmax, ymin, ymax, m, res=200000):
    """Return equal area grid in the specified map projection.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Boundaries of the grid
    m : Basemap object
        Map projection to convert between lat-lon and map coordinates. Should
        be an equal-area projection, e.g. 'nplaea'
    res : float, optional
        Grid resolution in metres.

    Returns
    -------
    grid : xarray.Dataset
        Grid points in map coordinates (X, Y) and lat-lon coordinates.
    """
    xi = np.arange(xmin, xmax + 1, res)
    yi = np.arange(ymin, ymax + 1, res)
    grid_XM, grid_YM = np.meshgrid(xi, yi)
    grid_X, grid_Y = convert_coords(grid_XM, grid_YM, m, inverse=True)
    dims = ['y', 'x']
    coords = {'x' : xi, 'y' : yi}
    grid = xr.Dataset({'lon' : xr.DataArray(grid_X, dims=dims, coords=coords),
                       'lat' : xr.DataArray(grid_Y, dims=dims, coords=coords),
                       'X' : xr.DataArray(grid_XM, dims=dims, coords=coords),
                       'Y' : xr.DataArray(grid_YM, dims=dims, coords=coords)})
    grid.attrs['resolution'] = res
    grid.attrs['map_info'] = str(get_map_info(m))
    return grid


def coastline_helper(resolution='c', ll_lat=None, ll_lon=None,
                  ur_lat=None, ur_lon=None):
    """Return coastlines in lat-lon for selected resolution.

    Coastline data is obtained from Basemap.drawcoastlines() in the
    basemap package.

    Parameters
    ----------
    resolution : {'c', 'l', 'i', 'h', 'f'}, optional
        Resolution options are 'c' (crude), 'l' (low), 'i' (intermediate),
        'h' (high), or 'f' (full)
    ll_lat, ll_lon, ur_lat, ur_lon : float, optional
        Lat/lon of lower-left corner (ll_lat, ll_lon) and upper-right corner
        (ur_lat, ur_lon) of area to extract data.
        If omitted, the global dataset is returned.

    Returns
    -------
    coastline : pd.DataFrame
        DataFrame of lat-lon coordinates of coastlines.
    """
    m = Basemap(projection='cyl', resolution=resolution, llcrnrlon=ll_lon,
                llcrnrlat=ll_lat, urcrnrlon=ur_lon, urcrnrlat=ur_lat)
    coast = m.drawcoastlines()
    plt.close()
    segments = coast.get_segments()
    nans = np.array([[np.nan, np.nan]])
    pts = segments[0]
    for seg in segments[1:]:
        pts = np.concatenate([pts, nans, seg], axis=0)
    coastline = pd.DataFrame(pts, columns=['lon', 'lat'])
    return coastline


def pcolor_map(var, xedge=None, yedge=None, coast=None, m=None, cmap='jet',
               colorbar=True, xname='x', yname='y', xlimits=None, ylimits=None,
               climits=None, meridians=range(-180, 181, 20),
               parallels=range(40, 81, 10),
               tick_fmt='%.0e', coast_kw={'color' : '0.3'},
               merid_para_kw={'linewidth' : 0.5, 'color' : '0.5'},
               cb_kw={'extend' : 'both'}):

    if m is None and (meridians is not None or parallels is not None):
        meridians, parallels = None, None
        print('Warning: input m must be provided to plot meridians and '
              'parallels')

    # If gridpoints are stacked, create a multiindex and unstack them
    if var.ndim == 1:
        dim = var.dims[0]
        var_plot = var.set_index(**{dim : [yname, xname]}).unstack(dim)
    else:
        var_plot = var

    if xedge is None:
        x = var[xname]
    else:
        x = xedge
    if yedge is None:
        y = var[yname]
    else:
        y = yedge
    if xlimits is None:
        xlimits = min(x), max(x)
    if ylimits is None:
        ylimits = min(y), max(y)

    plt.pcolormesh(x, y, var_plot, cmap=cmap)
    plt.gca().set_aspect('equal')
    if colorbar:
        plt.colorbar(**cb_kw)
    if climits is not None:
        plt.clim(climits)
    if coast is not None:
        plt.plot(coast[xname], coast[yname], **coast_kw)
    if meridians is not None or parallels is not None:
        plot_parallels_meridians(m, parallels=parallels, meridians=meridians,
                                 **merid_para_kw)
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    format_ticks(xfmt=tick_fmt, yfmt=tick_fmt)


def greenland_coast(coast, x0=6e6, y0=4e6, xname='x', yname='y'):
    """Return subset of coast dataframe corresponding to Greenland"""
    dsq = np.square(coast[xname] - x0) + np.square(coast[yname] - y0)
    i0 = dsq.argmin()
    isplit = np.where(np.isnan(coast[xname]))[0]
    istart = isplit[isplit <= i0][-1]
    ifinish = isplit[isplit >= i0][0]
    coast_sub = coast[istart:ifinish]
    return coast_sub


def inside_polygon(x, y, poly_x, poly_y):
    """Return indices of points inside the polygon"""
    def vecs_to_array(x, y):
        x = x.reshape((len(x), 1))
        y = y.reshape((len(y), 1))
        return np.concatenate([x, y], axis=1)

    if isinstance(x, xr.DataArray) or isinstance(x, pd.Series):
        x, y = x.values, y.values
    if isinstance(poly_x, xr.DataArray) or isinstance(poly_x, pd.Series):
        poly_x, poly_y = poly_x.values, poly_y.values

    polygon = vecs_to_array(poly_x, poly_y)
    points = vecs_to_array(x, y)
    path = Path(polygon)
    inside = path.contains_points(points)
    return inside
