import os

import numpy as np
import pandas as pd
import scipy
import random
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

from wapiti import wapiti_tools, wapiti

def standard_wapitify(time, rvs, drvs, frequency = np.linspace(1/1000, 1/1.1, 100000), regularization = 0):

	# Create masked arrays for the used RVs and RV uncertainties, masking any NaN values
	ma_rvs = np.ma.MaskedArray((rvs.T), mask=np.isnan((rvs.T)))
	ma_drvs = np.ma.MaskedArray((drvs.T), mask=np.isnan((drvs.T)))

	# Compute the average and variance of the used RVs and RV uncertainties using the masked arrays
	average = np.ma.average(ma_rvs, weights=1/ma_drvs**2, axis=1)
	variance = np.ma.average((ma_rvs-average.reshape(-1, 1))**2, weights=1/ma_drvs**2, axis=1)

	# Reshape the averages and standard deviations into column vectors
	mean_X = average.data.reshape(-1, 1)
	std_X = np.sqrt(variance.data.reshape(-1, 1))

	# Normalize the used RVs and RV uncertainties
	X = (np.copy(rvs.T)-mean_X)/std_X
	dX = np.copy(drvs.T)/std_X
	weights = 1/dX
	weights[np.isnan(X)] = 0 

	p_val = wapiti.compute_p_values(X, weights, regularization = regularization)

	n_components = 0
	while p_val[n_components] < 1e-5:
	    n_components += 1

	pearson_array = wapiti.lpocv(X, weights, n_components, regularization = regularization)
	pearson = np.nanmean(pearson_array, axis=1)
	n_components = 0
	while pearson[n_components] >= 0.95:
	    n_components += 1

	# Fit WPCA model
	pca = WPCA(n_components=n_components)
	pca.regularization = regularization  
	pca.fit(X, weights=weights)

	D = wapiti_tools.compute_anomaly_degree(pca)	

	faps = wapiti_tools.find_optimal_rejection(D, time, rvs, drvs, frequency, n_components)

	index_sort = np.argsort(D)[::-1]
	optimal_indx = np.argmin(faps)
	time_used = np.delete(time, index_sort[:optimal_indx+1])

	# Mask the copies of the arrays using the mask
	rvs_used = np.delete(np.copy(rvs), index_sort[:optimal_indx+1], axis=0)
	drvs_used = np.delete(np.copy(drvs) , index_sort[:optimal_indx+1], axis=0)

	# Create masked arrays for the used RVs and RV uncertainties, masking any NaN values
	ma_rvs = np.ma.MaskedArray((rvs_used.T), mask=np.isnan((rvs_used.T)))
	ma_drvs = np.ma.MaskedArray((drvs_used.T), mask=np.isnan((drvs_used.T)))

	# Compute the average and variance of the used RVs and RV uncertainties using the masked arrays
	average = np.ma.average(ma_rvs, weights=1/ma_drvs**2, axis=1)
	variance = np.ma.average((ma_rvs-average.reshape(-1, 1))**2, weights=1/ma_drvs**2, axis=1)

	# Reshape the averages and standard deviations into column vectors
	mean_X = average.data.reshape(-1, 1)
	std_X = np.sqrt(variance.data.reshape(-1, 1))

	# Normalize the used RVs and RV uncertainties
	X = (np.copy(rvs_used.T)-mean_X)/std_X
	dX = np.copy(drvs_used.T)/std_X

	# Compute weights for the RVs based on the normalized RV uncertainties
	weights = 1. / dX
	weights[np.isnan(X)] = 0 

	pca = WPCA(n_components=n_components)
	pca.regularization = regularization
	pca.fit(X, weights=weights)

	rv_0, std_rv_0 = [], []
	for idx in tqdm(range(len(time_used)), leave=False):
		rv_temp, std_rv_temp = wapiti_tools.odd_ratio_mean(rvs_used[idx], drvs_used[idx])
		rv_0.append(rv_temp)
		std_rv_0.append(std_rv_temp)
	rv_0, std_rv_0 = np.array(rv_0), np.array(std_rv_0)
	average, std = wapiti_tools.odd_ratio_mean(rv_0, std_rv_0)
	wpca_model = pca.reconstruct([(rv_0 - average)/std], weights=[std/std_rv_0])

	rv_wapiti = (((rv_0 - average)/std - wpca_model)*std + average)[0]
	std_rv_wapiti = std_rv_0

	results_wapiti =  {}
	results_wapiti['time_used'] = time_used
	results_wapiti['rv_wapiti'] = rv_wapiti
	results_wapiti['std_rv_wapiti'] = std_rv_wapiti
	results_wapiti['p_val'] = p_val 
	results_wapiti['pearson'] = pearson
	results_wapiti['n_components'] = n_components
	results_wapiti['D'] = D
	results_wapiti['faps'] = faps

	return results_wapiti

def wapitify(time, rvs, drvs, frequency = np.linspace(1/1000, 1/1.1, 100000), n_components = None, regularization = 0):

	if n_components is None:

		results_wapiti = standard_wapitify(time, rvs, drvs, frequency = np.linspace(1/1000, 1/1.1, 100000), regularization = regularization)
	
	else:
		# Create masked arrays for the used RVs and RV uncertainties, masking any NaN values
		ma_rvs = np.ma.MaskedArray((rvs.T), mask=np.isnan((rvs.T)))
		ma_drvs = np.ma.MaskedArray((drvs.T), mask=np.isnan((drvs.T)))

		# Compute the average and variance of the used RVs and RV uncertainties using the masked arrays
		average = np.ma.average(ma_rvs, weights=1/ma_drvs**2, axis=1)
		variance = np.ma.average((ma_rvs-average.reshape(-1, 1))**2, weights=1/ma_drvs**2, axis=1)

		# Reshape the averages and standard deviations into column vectors
		mean_X = average.data.reshape(-1, 1)
		std_X = np.sqrt(variance.data.reshape(-1, 1))

		# Normalize the used RVs and RV uncertainties
		X = (np.copy(rvs.T)-mean_X)/std_X
		dX = np.copy(drvs.T)/std_X
		weights = 1/dX
		weights[np.isnan(X)] = 0 

		pca = WPCA(n_components=n_components)
		pca.regularization = regularization
		pca.fit(X, weights=weights)


		rv_0, std_rv_0 = [], []
		for idx in tqdm(range(len(time)), leave=False):
			rv_temp, std_rv_temp = wapiti_tools.odd_ratio_mean(rvs[idx], drvs[idx])
			rv_0.append(rv_temp)
			std_rv_0.append(std_rv_temp)
		rv_0, std_rv_0 = np.array(rv_0), np.array(std_rv_0)
		average, std = wapiti_tools.odd_ratio_mean(rv_0, std_rv_0)
		wpca_model = pca.reconstruct([(rv_0 - average)/std], weights=[std/std_rv_0])
		
		rv_wapiti = (((rv_0 - average)/std - wpca_model)*std + average)[0]
		std_rv_wapiti = std_rv_0

		results_wapiti =  {}
		results_wapiti['time'] = time
		results_wapiti['rv_wapiti'] = rv_wapiti
		results_wapiti['std_rv_wapiti'] = std_rv_wapiti
		results_wapiti['n_components'] = n_components
	
	return results_wapiti