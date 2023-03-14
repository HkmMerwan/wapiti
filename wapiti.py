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

def permute_and_fit_pca(X, weights, n, n_components):
    """Permute the columns of X and weights and fit a WPCA model.

    Parameters
    ----------
    X : numpy array
        Array with shape (n_samples, n_features) containing the data.
    weights : numpy array
        Array with shape (n_samples, n_features) containing the weights.
    n : int
        Number of data points.
    n_components : int
        Number of principal components to keep.

    Returns
    -------
    pca : WPCA object
        Fitted WPCA model.
    """
    X_sim = np.copy(X)
    weights_sim = np.copy(weights)
    for col in range(X.shape[1]):
        # Permute columns of X and weights
        indexes = random.sample(range(n), n)
        X_sim[:, col] = X_sim[indexes, col]
        weights_sim[:, col] = weights_sim[indexes, col]
    pca = WPCA(n_components=n_components)
    pca.regularization = 2  # Fix regularization at 2
    pca.fit(X_sim, weights=weights_sim)
    return pca


def compute_variance_ratios(X, weights, n_components, N_permutations):
    """Compute explained variance ratios for permuted WPCA models.

    Parameters
    ----------
    X : numpy array
        Array with shape (n_samples, n_features) containing the data.
    weights : numpy array
        Array with shape (n_samples, n_features) containing the weights.
    n_components : int
        Number of principal components to keep.
    N_permutations : int
        Number of permutations to perform.

    Returns
    -------
    variance : numpy array
        Array with shape (N_permutations, n_features) containing the
        explained variance ratios for each permuted model.
    """
    variance = np.zeros((N_permutations, X.shape[1]))
    n = X.shape[0]
    for i in tqdm(range(N_permutations), leave=False):
        pca = permute_and_fit_pca(X, weights, n, n_components)
        variance[i] = pca.explained_variance_ratio_
    return variance


def compute_p_values(X, weights, N_permutations=100):
    """Compute p-values for a WPCA model using a permutation test.

    Parameters
    ----------
    X : numpy array
        Array with shape (n_samples, n_features) containing the data.
    weights : numpy array
        Array with shape (n_samples, n_features) containing the weights.
    N_permutations : int, optional
        Number of permutations to perform. Default is 100.

    Returns
    -------
    p_val : numpy array
        Array with shape (n_features,) containing the p-values for each
        feature.
    """
    # Preprocessing steps
    n_components = X.shape[1]
    
    # Fit non-permuted WPCA model
    pca_0 = WPCA(n_components=n_components)
    pca_0.regularization = 2  # Fix regularization at 2
    pca_0.fit(X, weights=weights)
    
    # Compute explained variance ratios for permuted models
    variance = compute_variance_ratios(X, weights, n_components, N_permutations)
    
    # Compute p-values from explained variance ratios
    p_val = np.sum(variance > pca_0.explained_variance_ratio_, axis=0) / N_permutations
    return p_val

def lpocv(X, weights, n_components, p=.8, N=100):
    """
    Compute the Pearson correlation coefficient for the principal components.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data matrix.
    weights : array-like of shape (n_samples,)
        The weight of each sample.
    n_components : int
        The number of principal components to compute.
    p : float, default=.8
        The fraction of samples to use for each PCA.
    N : int, default=100
        The number of permutations to perform.

    Returns
    -------
    pearson : array of shape (n_components, N)
        The Pearson correlation coefficient for each principal component.
    """
    # Fit non-permuted PCA model
    pca_0 = WPCA(n_components=n_components)
    pca_0.regularization = 2  # Fix regularization at 2
    pca_0.fit(X, weights=weights)
    
    pca_0_components = pca_0.components_
    n = int(p*X.shape[0])

    pca_components_array = [pca_0_components]
    for _ in tqdm(range(N)):
        # Sample a subset of the data
        n = int(p*X.shape[0])
        indexes = random.sample(range(X.shape[0]), n)
        boolean_indexes = []
        for idx in range(X.shape[0]):
            if idx in indexes:
                boolean_indexes.append(True)
            else:
                boolean_indexes.append(False)
        boolean_indexes = np.array(boolean_indexes)
        
        X_sample = X[boolean_indexes]
        weights_sample = weights[boolean_indexes]
            
        # Construct the PCA model on the subset of the data
        pca = WPCA(n_components=n_components)
        pca.regularization = 2  # Fix regularization at 2
        pca.fit(X_sample, weights=weights_sample)

        pca_components = pca.components_
        pca_components_array.append(pca_components)

    pca_components_array = np.array(pca_components_array)
    
    pearson = []
    for i in range(n_components):
        pearson_temp = []
        for j in range(N):
            # Compute Pearson correlation coefficient for each principal component
            value = np.abs(scipy.stats.pearsonr(pca_components_array[0, i], pca_components_array[j+1, i])[0])
            pearson_temp.append(value)
        pearson.append(np.array(pearson_temp))
    pearson = np.array(pearson)
    
    return pearson
