import os

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.optimize import minimize

from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import LombScargle

import celerite
from celerite.modeling import Model
from celerite import terms

import emcee

from wpca import WPCA

import radvel

from matplotlib import pyplot as plt
import seaborn

from tqdm.notebook import tqdm as tqdm

seaborn.set_context("notebook")
plt.rcParams["axes.grid"] = False
cwd = os.getcwd()  # current working directory


def absolute_deviation(v: np.ndarray) -> np.ndarray:

    """
    Parameters
    ----------
    v: float array
        > Input array.
        
    Returns
    -------
    AD: float array
        > The absolute deviation of the input array
    """
    
    v_med=np.nanmedian(v)
    AD=(np.abs(v-v_med))
    return AD

def mad(v: np.ndarray) -> float:

    """
    Parameters
    ----------
    v: float array
        > Input array.
        
    Returns
    -------
    MAD: float array
        > The median absolute deviation of the input array
    """
    
    v_med=np.nanmedian(v)
    MAD=np.nanmedian(absolute_deviation(v))
    return MAD

def remove_outliers(k: float, v: np.ndarray, *params) -> tuple:

    """
    Parameters
    ----------
    
    k: float
        > k-clipping    
    v: float array
        > Input array which outliers are to be removed
    params: float array(s)
        > Other arrays
        
    Returns
    -------
    res: tuple of np.ndarrays
        > arrays with index removed corresponding to outilers of v
    """
    
    
    madv=mad(v)
    keep_index = absolute_deviation(v) < k*madv
    res = [v[keep_index]]

    for param in params:
        res.append(param[keep_index])
        
    return tuple(res)

def odd_ratio_mean(value, err, odd_ratio=1e-4, nmax=10):
    # Vectorized implementation of odd_ratio_mean
    
    # Mask NaNs
    mask = np.isfinite(value) & np.isfinite(err)
    if not np.any(mask):
        return np.nan, np.nan

    # Apply mask
    value = value[mask]
    err = err[mask]

    # Initial guess
    guess = np.nanmedian(value)

    for nite in range(nmax):
        nsig = (value - guess) / err
        gg = np.exp(-0.5 * nsig**2)
        odd_bad = odd_ratio / (gg + odd_ratio)
        odd_good = 1 - odd_bad
        w = odd_good / err**2
        guess = np.nansum(value * w) / np.nansum(w)

    bulk_error = np.sqrt(1 / np.nansum(odd_good / err**2))

    return guess, bulk_error

def night_bin(times, rv, drv=None, binsize=0.5):
    
    """
    Bin data by night, using a moving window of size `binsize`.
    
    Parameters
    ----------
    times : numpy.ndarray
        The time array.
    rv : numpy.ndarray
        The RV array.
    drv : numpy.ndarray, optional
        The error on the RV array. If provided, the weighted mean and variance of the RV values will be
        calculated. If not provided, the unweighted mean and variance will be calculated.
    binsize : float, optional
        The size of the moving window, in days.
    
    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
       The time, RV, and error on RV arrays for the binned data.
    """
    
    n = len(rv)
    res_times = np.empty(n)
    res_rv = np.empty(n)
    res_drv = np.empty(n)

    times_temp = []
    rv_temp = []
    drv_temp = []

    time_0 = times[0]

    res_index = 0
    for index in range(n):
        if np.abs(times[index] - time_0) <= binsize:
            times_temp.append(times[index])
            rv_temp.append(rv[index])
            if drv is not None:
                drv_temp.append(drv[index])

        else:
            
            times_temp = np.array(times_temp)
            res_times[res_index] = times_temp.mean()

            rv_temp = np.array(rv_temp)
            if drv is not None:
                drv_temp = np.array(drv_temp)

            mask = ~np.isnan(rv_temp)
        
            if mask.sum() > 0:
                if drv is not None:
                    weights = 1 / (drv_temp[mask]) ** 2
                    weights /= weights.sum()
                    average = (weights * rv_temp[mask]).sum()
                    var = (weights ** 2 * (drv_temp[mask]) ** 2).sum()
                    res_rv[res_index] = average
                    res_drv[res_index] = np.sqrt(var)
                else:
                    res_rv[res_index] = rv_temp[mask].mean()
                    res_drv[res_index] = rv_temp[mask].std()

                res_index += 1
            else:
                res_rv[res_index] = np.nan
                res_drv[res_index] = np.nan
                res_index += 1
                

            time_0 = times[index]
            times_temp = [time_0]
            rv_temp = [rv[index]]
            if drv is not None:
                drv_temp = [drv[index]]

    if drv is not None:
        return res_times[:res_index], res_rv[:res_index], res_drv[:res_index]
    else:
        return res_times[:res_index], res_rv[:res_index]

def compute_anomaly_degree(pca):
    """
    Compute the anomaly degree of each epoch based on the maximum ratio between the absolute deviation and the median 
    absolute deviation among all principal vectors V_n. This is used to identify anomalous observations.
    
    Parameters:
        pca (WPCA): A fitted wPCA model containing principal components.
    
    Returns:
        numpy.ndarray: An array of size (n_epochs,) containing the anomaly degree of each epoch.
    """    
    # Initialize distance_mad array to store the ratio of absolute deviation to median absolute deviation
    distance_mad = np.zeros((pca.components_.shape[1], pca.components_.shape[0]))

    for idx in range(pca.components_.shape[0]):
        # Calculate MAD of current component
        madv = mad(pca.components_[idx])
        # Calculate absolute deviation of current component
        ad = absolute_deviation(pca.components_[idx])
        # Store ratio of absolute deviation to median absolute deviation
        distance_mad[:, idx] = ad/madv

    # Get the maximum value of the distance_mad array along the rows
    D = np.max(distance_mad, axis=1)
    
    return D

def find_optimal_rejection(D, time_binned, rvs_binned, drvs_binned, frequency, n_components, regularization=0):
    """
    This function computes the false alarm probabilities (FAPs) for each set of RV data after removing epochs one by one,
    sorted in decreasing order of their anomaly degree D. 

    Args:
    - D: array of anomaly degree
    - time_binned: array of binned time values
    - rvs_binned: array of binned RV values
    - drvs_binned: array of binned RV uncertainty values
    - frequency: array of angular frequencies at which to compute the Lomb-Scargle periodogram
    - n_components: number of principal components to retain for the WPCA analysis (default=20)

    Returns:
    - faps: array of FAPs for each set of RV data after removing epochs one by one
    """

    # Sort the epochs by decreasing D value
    index_sort = np.argsort(D)[::-1]

    faps = []
    d_idx = 0
    while D[index_sort][d_idx] >= 5:
        # Remove the epoch with the highest D value
        time_used = np.delete(time_binned, index_sort[:d_idx+1])
        rvs_used = np.copy(rvs_binned)
        drvs_used = np.copy(drvs_binned)
        rvs_used = np.delete(rvs_used, index_sort[:d_idx+1], axis=0)
        drvs_used = np.delete(drvs_used , index_sort[:d_idx+1], axis=0)
        rvs_used = rvs_used.T
        drvs_used = drvs_used.T

        # Compute the weighted average and variance of the remaining RV data
        ma_rvs = np.ma.MaskedArray((rvs_used), mask=np.isnan((rvs_used)))
        ma_drvs = np.ma.MaskedArray((drvs_used), mask=np.isnan((drvs_used)))
        average = np.ma.average(ma_rvs, weights=1/ma_drvs**2, axis=1)
        variance = np.ma.average((ma_rvs-average.reshape(-1, 1))**2, weights=1/ma_drvs**2, axis=1)
        mean_X = average.data.reshape(-1, 1)
        std_X = np.sqrt(variance.data.reshape(-1, 1))

        # Normalize the RVs and RV uncertainties
        rv = (np.copy(rvs_used)-mean_X)/std_X
        drv = np.copy(drvs_used)/std_X

        # Compute weights for the RVs based on the normalized RV uncertainties
        weights = 1. / drv
        weights[np.isnan(rv)] = 0 

        # Fit a WPCA model to the normalized RVs and weights
        pca_0 = WPCA(n_components=n_components)
        pca_0.regularization = regularization
        pca_0.fit(rv, weights=weights)
        wpca_model = pca_0.reconstruct(rv, weights=weights)

        rvs_used = ((rv - wpca_model)*std_X + mean_X).T
        rv_used, std_rv_used = [], []
        for idx in tqdm(range(len(time_used)), leave=False):
            rv_temp, std_rv_temp = odd_ratio_mean(rvs_used[idx], drvs_used.T[idx])
            rv_used.append(rv_temp)
            std_rv_used.append(std_rv_temp)
        rv_used, std_rv_used = np.array(rv_used), np.array(std_rv_used)

      
        # LombScargle
        ls = LombScargle(time_used, rv_used, std_rv_used)
        power = ls.power(frequency)
        fap = ls.false_alarm_probability(power.max())
        faps.append(fap)
        
        d_idx += 1
        
    faps = np.array(faps)

    return faps