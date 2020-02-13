import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr


def find_missings(data, verbose=True):
    """Return a list or dict of lists of indices with missing data.

    Input 'data' is pd.Series or pd.DataFrame.
    """
    missings = {}
    if isinstance(data, pd.Series):
        series_flag = True
        if data.name is not None:
            series_name = data.name
        df = data.to_frame(name=series_name)
    else:
        df = data
        series_flag, series_name = False, None
    for col in df.columns:
        missings[col] = df.index[df[col].isnull()]
        if verbose:
            print(col + ': %d missings' % len(missings[col]))
    if series_flag:
        missings = missings[series_name]
    return missings


def subset(data, subset_dict, incl_lower=True, incl_upper=True,
           copy=True, apply_squeeze=False):
    """Extract a subset of xarray DataArray or Dataset along named dimensions.

    Returns a DataArray or Dataset sub extracted from input data,
    such that:
        sub[dim_name] >= lower_or_list & sub[dim_name] <= upper,
    OR  sub[dim_name] == lower_or_list (if lower_or_list is a list)
    for each dim_name in subset_dict.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data source for extraction.
    subset_dict : dict of 2-tuples
        Dimensions and subsets to extract.  Each entry in subset_dict
        is in the form {dim_name : (lower_or_list, upper)}, where:
        - dim_name : string
            Name of dimension to extract from.
        - lower_or_list : scalar or list of int or float
            If scalar, then used as the lower bound for the   subset range.
            If list, then the subset matching the list will be extracted.
        - upper : int, float, or None
            Upper bound for subset range. If lower_or_list is a list,
            then upper is ignored and should be set to None.
    incl_lower, incl_upper : bool, optional
        If True lower / upper bound is inclusive, with >= or <=.
        If False, lower / upper bound is exclusive with > or <.
        If lower_or_list is a list, then the whole list is included
        and these parameters are ignored.
    copy : bool, optional
        If True, return a copy of the data, otherwise return a pointer.
    apply_squeeze : bool, optional
        If True, squeeze out any singleton dimensions.

    Returns
    -------
        sub : xr.DataArray or xr.Dataset
    """

    def subset_1dim(data, dim_name, lower_or_list, upper=None,
                     incl_lower=True, incl_upper=True, copy=True):
        """Extract a subset of a DataArray along a named dimension."""
        vals = data[dim_name]
        if upper is None:
            valrange = lower_or_list
        else:
            if incl_lower:
                ind1 = vals >= lower_or_list
            else:
                ind1 = vals > lower_or_list
            if incl_upper:
                ind2 = vals <= upper
            else:
                ind2 = vals < upper
            valrange = vals[ind1 & ind2]
        data_out = data.sel(**{dim_name : valrange})
        if copy:
            data_out = data_out.copy()
        return data_out

    sub = data
    for dim_name in subset_dict:
        lower_or_list, upper = subset_dict[dim_name]
        sub = subset_1dim(sub, dim_name, lower_or_list, upper, incl_lower,
                          incl_upper, copy)

    if apply_squeeze:
        sub = squeeze(sub)

    return sub


def readmat_struct(filenm, struct_nm, squeeze=True, dataframe=True,
                   indexnm=None):
    """Read Matlab struct from .mat file and return as dict or DataFrame"""
    data = sio.loadmat(filenm)
    nms = data[struct_nm].dtype.names
    output = {}
    for nm in nms:
        vals = data[struct_nm][nm][0, 0]
        if squeeze:
            vals = vals.squeeze()
        output[nm] = vals
    if dataframe:
        if indexnm is None:
            index = range(len(output[nms[0]]))
        else:
            index = output[indexnm]
            output.pop(indexnm)
        output = pd.DataFrame(output, index=index)
    return output
