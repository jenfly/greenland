"""Surface energy balance model"""

import numpy as np
import pandas as pd


def melt_model(seb, surface_temp_col='Ts', net_seb_col='SEB_NET',                     
               time_res='hourly'):
    """Return a Series of melt rates estimated from surface energy balance.
    
    
    
    Parameters
    ----------
    seb : pd.DataFrame
        DataFrame including the surface temperature in deg C and net surface
        energy balance in W/m2.
    surface_temp_col, net_seb_col : str, optional
        Column names for surface temperature and net SEB in the input DataFrame.
    time_res : {'hourly', 'daily'}, optional
        Time resolution of input energy balance data. Used to convert from W/m2 
        to melt in cm w.e.
        
    Returns
    -------
    melt : pd.Series
        Estimated melt rates in cm w.e. per hour (for hourly data) or
        cm w.e. per day (for daily data).
    """

    # Set melt energy to zero wherever surface temperature < 0 or net SEB < 0
    frozen = seb[surface_temp_col] < 0
    neg_net_seb = seb[net_seb_col] < 0
    melt_energy = seb[net_seb_col].copy()
    melt_energy[frozen | neg_net_seb] = 0
 
    # Estimate melt in cm w.e.
    lh_fusion = 334e3 # J/kg
    rho_water = 997 # kg / m3
    s_perhour = 60 * 60
    cm_perm = 100
    melt = (melt_energy * s_perhour * cm_perm / (lh_fusion * rho_water))
    if time_res.lower() == 'daily':
        melt = 24 * melt

    return melt


def sublimation_model(seb, surface_temp_col='Ts', lhf_col='LHF',
                      time_res='hourly'):
    """Return Series of sublimation / resublimation from surface energy balance.
    
    Values are positive (sublimation) when LHF < 0 (any Ts).
    Values are negative (resublimation) when LHF > 0 and Ts < 0.
    Values are 0 when LHF > 0 and Ts = 0 (condensation).
    
    Input `time_res` (either 'hourly' or 'daily') is used to convert from W/m2 
    to melt in cm w.e / hour or cm w.e. /day.  
    """
    
    # Estimate sublimation in cm w.e.
    lh_sublimation = 2.849e6 # J/kg
    rho_water = 997 # kg / m3
    s_perhour = 60 * 60
    cm_perm = 100
    sublimation = - (seb[lhf_col] * s_perhour * cm_perm / 
                     (lh_sublimation * rho_water))
    if time_res.lower() == 'daily':
        sublimation = 24 * sublimation
        
    # When LHF > 0 and Ts = 0 we have condensation instead of sublimation or
    # resublimation, so set these values to 0
    condensation = (seb[lhf_col] > 0) & (seb[surface_temp_col] == 0)
    sublimation.loc[condensation] = 0

    return sublimation


def ablation_model(seb, surface_temp_col='Ts', net_seb_col='SEB_NET', lhf_col='LHF',
                   time_res='hourly'):
    """Return DataFrame with estimated melt, sublimation, and ablation.

    Melt, sublimation and ablation are in cm w.e./hr or cm w.e./day, depending on
    the time resolution of the input data.
    
    Energy budget components in the input seb DataFrame must be in W/m2.
    
    Input `time_res` (either 'hourly' or 'daily') is used to convert from W/m2 
    to melt in cm w.e.
    """
    
    melt = melt_model(seb, surface_temp_col=surface_temp_col, 
                         net_seb_col=net_seb_col, time_res=time_res)
    output = melt.to_frame(name='melt')
    output['sublimation'] = sublimation_model(seb, surface_temp_col=surface_temp_col, 
                                              lhf_col=lhf_col, time_res=time_res)
    
    # Use skipna=True so that if melt=0 and sublimation is missing, ablation is set to 0
    # instead of NaN
    output['ablation'] = output[['melt', 'sublimation']].sum(axis=1, skipna=True)
    
    # Set ablation to NaN anywhere that melt is missing
    output.loc[output['melt'].isnull(), 'ablation'] = np.nan
    
    return output
    