import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import xarray as xr

from . import munge

def standardize(data, cols=None):
    """Return standardized pd.Series or pd.DataFrame with standardized columns.

    Input data is either a pd.Series or pd.DataFrame.  If DataFrame, a subset
    of columns can be specified with the `cols` input.  If omitted, all columns
    are standardized.

    Columns are standardized as: y_out = (y - y.mean()) / y.std()
    """
    def standardize_one(series):
        return (series - series.mean()) / series.std()

    if isinstance(data, pd.Series):
        data_out =  standardize_one(data)
    elif isinstance(data, pd.DataFrame):
        data_out = data.copy()
        if cols is None:
            cols = data.columns
        data_out[cols] = data_out[cols].apply(standardize_one)
    else:
        raise ValueError('Input data must be pd.Series or pd.DataFrame')
    return data_out


def mlr(formula='Y ~ X', data=None, verbose=True):
    """Multiple linear regression model"""
    lm = smf.ols(formula=formula, data=data).fit()
    Y_regr = lm.predict()
    if verbose:
        print(lm.summary())
    return lm, Y_regr


def princomp(y, kmax=None, eigenrows=True, real=True):
    """Perform principal component analysis on matrix y.

    Uses numpy functions to compute eigenvalues and eigenvectors.

    Parameters
    ----------
    y : np.array (2-dimensional)
        Input data matrix.  Rows are observations and columns are variables.
    kmax : int, optional
        Truncates at first kmax modes (or returns all modes if kmax is None).
    eigenrows : bool, optional
        If eigenrows is True (default), eigenvectors and principal components
        are output as rows of their respective matrices, and the data y can be
        reconstructed as:
          y_rec = y_mean + np.dot(A, E)
          where A = output['scores'] is the matrix of principal components and
          E = output['eigenvec'] is the matrix of eigenvectors.
        If eigenrows is False, then y_rec = y_mean + np.dot(A.T, E.T)
    real : bool, optional
        If True, return eigenvectors, eigenvalues, etc. as real, otherwise
        return complex arrays.

    Output
    ------
    pca : dict
        Dictionary of eigenvectors, eigenvalues, principal components (scores),
        fraction of variance for each mode, original data (y_orig), and
        reconstructed data (y_rec).
    """

    # Subtract the mean (along columns) and transpose
    mean_y = np.mean(y.T, axis=1)
    yn = (y - mean_y).T

    # Compute the covariance matrix
    s = np.cov(yn)

    # Compute eigenvalues and eigenvectors and sort descending
    eigval, eigvec = np.linalg.eig(s)
    if real:
        eigval, eigvec = eigval.real, eigvec.real
    idx = np.argsort(eigval)
    idx = idx[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]

    # Transpose eigvec so each row is an eigenvector
    eigvec = eigvec.T

    # Fraction of variance explained by each mode
    variance_all = eigval / eigval.sum()

    # Truncate at kmax
    if kmax is not None:
        eigval = eigval[:kmax]
        eigvec = eigvec[:kmax]
        variance = variance[:kmax]
    else:
        variance = variance_all

    # Compute principal component scores
    a = np.dot(eigvec, yn)
    a = a.T

    # Reconstruct y from the eigenvectors and principal components
    y_rec = mean_y + np.dot(a, eigvec)

    if not eigenrows:
        eigvec, a = eigvec.T, a.T

    pca = {'eigval' : eigval, 'eigvec' : eigvec, 'varfrac' : variance,
           'varfrac_all' : variance_all, 'scores' : a, 'y_rec' : y_rec,
           'y_orig' : y}

    return pca


def pca_xr(data_in, kmax=None, real=True):
    """xarray wrapper for princomp()"""

    pca = princomp(data_in.values, kmax=kmax, real=real, eigenrows=True)
    eigval, eigvec = pca['eigval'], pca['eigvec']

    pca_ds = xr.Dataset({'data_in' : data_in})

    # Eigenvalues and fraction of variance
    coords = {'mode' : np.arange(1, len(eigval) + 1)}
    pca_ds['eigval'] = xr.DataArray(eigval, dims=['mode'], coords=coords)
    pca_ds['varfrac'] = xr.DataArray(pca['varfrac'], dims=['mode'], coords=coords)
    mode_all = {'mode_all' : np.arange(1, len(pca['varfrac_all']) + 1)}
    pca_ds['varfrac_all'] = xr.DataArray(pca['varfrac_all'], dims=['mode_all'],
                                         coords=mode_all)

    # Eigenvectors DataArray
    dims = data_in.dims
    # --- Coordinates of input data columns
    coords = {nm : var for nm, var in data_in.coords.items() if dims[1] in var.dims}
    coords['mode'] = pca_ds['mode']
    pca_ds['eigvec'] = xr.DataArray(eigvec, dims=['mode', dims[1]], coords=coords)

    # PCs DataArray
    # --- Coordinates of input data rows
    coords = {nm : var for nm, var in data_in.coords.items() if dims[0] in var.dims}
    coords['mode'] = pca_ds['mode']
    pc = xr.DataArray(pca['scores'], dims=[dims[0], 'mode'], coords=coords)
    pc.attrs['long_name'] = 'Principal components'
    pca_ds['pc'] = pc

    # Reconstructed data
    pca_ds['data_rec'] = xr.DataArray(pca['y_rec'], dims=data_in.dims,
                                      coords=data_in.coords,
                                      attrs={'long_name' : 'Reconstructed data'})

    return pca_ds


def load_som(somfile, datafile, unstack_time=False):
    """Return xr.Dataset of SOM output from Matlab, with metadata.

    Input somfile is a .mat file with SOM output from Matlab.
    Input datafile is a .nc file that was the input data to the SOM analysis.
    """

    # Load the SOM input data
    print('Loading ' + datafile)
    with xr.open_dataset(datafile) as data:
        data.load()

    # Load the SOM output from Matlab and add metadata
    print('Loading ' + somfile)
    som_in = munge.readmat_struct(somfile, 'output', dataframe=False)
    attrs = {nm : int(som_in[nm]) for nm in ['ny_som', 'nx_som']}
    for nm in ['qe', 'te']:
        if nm in som_in:
            attrs[nm] = float(som_in[nm])
    if 'neigh' in som_in:
        attrs['neigh'] = str(som_in['neigh'])
    som = xr.Dataset(attrs=attrs)

    # SOM nodes
    if 'var' in data.dims:
        dim_nm = 'var'
    else:
        dim_nm = 'pt'
    dims = ['inode', dim_nm]
    coords = {nm : val for nm, val in data.coords.items() if dim_nm in val.dims}
    coords['inode'] = range(som_in['codebook'].shape[0])
    som['nodes'] = xr.DataArray(som_in['codebook'], dims=dims, coords=coords)

    # Best matching units and hits
    # --- Index node numbers from zero
    bmus = som_in['bmus'].astype(int) - 1
    coords = {nm : val for nm, val in data.coords.items() if 'time' in val.dims}
    som['bmus'] = xr.DataArray(bmus, dims=['time'], coords=coords)
    som['hits'] = xr.DataArray(som_in['hits'], dims=['inode'])
    som.coords['year'] = som['time'].dt.year
    som.coords['time_dt'] = som['time']
    som = som.set_index(time=['year', 'jday'])
    if unstack_time:
        som = som.unstack('time')


    # Arrange the nodes on a grid
    ny, nx = som.attrs['ny_som'], som.attrs['nx_som']
    som_grid = som['inode'].values.reshape((ny, nx), order='F')
    rows, cols = [], []
    for inode in som['inode'].values:
        pos = np.where(som_grid == inode)
        row, col = int(pos[0]), int(pos[1])
        rows.append(row)
        cols.append(col)
    som.coords['node_num'] = som['inode']
    som.coords['row'] = xr.DataArray(rows, dims=['inode'],
                                     coords={'inode' : som['inode']})
    som.coords['col'] = xr.DataArray(cols, dims=['inode'],
                                     coords={'inode' : som['inode']})
    som = som.set_index(inode=['row', 'col'])

    return som
