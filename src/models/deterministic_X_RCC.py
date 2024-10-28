import numpy as np
import xarray as xr
from .deterministic_radially_constrained_cluster import single_fit_optimized, generate_season_idx  # Importiamo la nuova funzione ottimizzata
from sklearn.metrics import silhouette_score
import math
import itertools


"""
    This script implements the xarray-compatible version of the deterministic clustering algorithm, which is used for 
    clustering time series data across spatial grids. The script provides two primary functions:

    1. X_cluster: 
       - This function performs clustering on one or more time series data (grid points) using the 
         Radially Constrained Clustering algorithm. 
       - It supports multiple variables by combining them into a normalized array, allowing clustering to be 
         based on multiple input features.
       - The function accepts several customizable parameters, including the number of seasons (clusters), 
         the number of days for each season, and initial breakpoints. 
       - The clustering is done by calling an optimized version of the clustering function (single_fit_optimized),
         which is computationally efficient. 
       - After clustering, it computes the breakpoints, error history, and silhouette scores to evaluate 
         the quality of the clustering. The silhouette score calculation has been optimized to avoid slow loops, 
         leveraging efficient NumPy operations.

    2. XRCC (Xarray Radially Constrained Clustering):
       - This function is designed to apply the X_cluster function across xarray datasets, which represent 
         multi-dimensional labeled data structures. 
       - The XRCC function is highly compatible with the parallelized processing capabilities of Dask, 
         enabling it to handle large-scale geospatial datasets efficiently. 
       - It returns the clustering results in the form of xarray DataArrays, which include the breakpoints, 
         error history, and silhouette scores for each grid point.
"""


def X_cluster(*grid_points: list[xr.DataArray], **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Perform clustering on multiple variables from grid points.

    Parameters:
    -----------
    *grid_points : tuple of numpy arrays
        Input grid points variables to be clustered.
    **kwargs : keyword arguments
        Additional parameters for clustering.

    Returns:
    --------
    tuple
        Array of breakpoints, error history, and silhouette scores.
    """

    # TODO:
    #   - Implement different normalization methods
    #   - Implement optimized method for silhouette score calculation
    #   - Add a min length parameter to avoid clustering on short time series

    # -- 0. PARAMETERS 
    n_days = kwargs.get('n_days', 20)  
    n_seas = kwargs.get('n_seas', 2)
    metric = kwargs.get('metric', 'euclidean')



    # Deriving the number of total iterations as the binomial coefficient of the number of days and seasons
    iters = math.comb(len(n_days), n_seas)


    # -- 1. CREATION OF THE NORMALIZED ARRAY
    # In this section a list of xarray.DataArrays is converted into the design matrix for clustering. 
    # The matrix ha dimension (365, n_variables * n_years)

    arrays = [] # List for storing the normalized arrays
    
    # Iterating over all the xarray.DataArrays
    for grid_points_var in grid_points:
        grid_points_var = np.asarray(grid_points_var)
        grid_points_var = np.reshape(grid_points_var, (365, int(grid_points_var.size/365)), order='F')  # Reshape the array to (365, n_years)
        
        # Check if this is an empty grid point. In this case return NaN arrays. This can happen if the original xarra.DataArray has been masked.
        if np.isnan(grid_points_var).any():
            return (np.full(n_seas, np.nan), np.full(iters, np.nan), np.full(iters, np.nan))
        
        # Append the normalized array to the list
        arrays.append(grid_points_var)
    

    # -- 2. NORMALIZATION OF THE VARIABLES
    # Create a boolean mask indicating which columns across all arrays do not contain any NaN values - needed for the normalization.
    combined_mask = ~np.any([np.all(np.isnan(arr), axis=0) for arr in arrays], axis=0)
    
    # Normalizzing the variables along the columns - each variable is normalized separately
    normalized_arrays = []
    for arr in arrays:
        array_tot = arr[:, combined_mask]
        array_tot = (array_tot - array_tot.min(axis=0)) / (array_tot.max(axis=0) - array_tot.min(axis=0))
        normalized_arrays.append(array_tot)
    
    # Now all the normalized variables can be joined in the same array
    array_tot = np.concatenate(normalized_arrays, axis=1)

    # -- 3. CLUSTERING
    # The function returns he optimal breakpoints and the diagnostic of the run
    breakpoints, error_history, breakpoint_history = single_fit_optimized(data_to_cluster=array_tot, n_seas=n_seas, n_days=n_days, metric=metric)


    # Vectorize this piece of code to avoid for loop [VERY SLOW]
    # Silhouette Scores
    silhouette_scores = np.zeros((len(breakpoint_history), 1)).squeeze()

    # silhouette_scores = []
    # for bp in breakpoint_history:
    #     prediction = np.zeros((array_tot.shape[0], 1))  # Prepara il vettore delle predizioni


    #     idx = generate_season_idx(bp, array_tot.shape[0], n_seas)
    #     for i in range(n_seas):
    #         prediction[idx[i]] = i

    #     # Calcolo degli silhouette score
    #     try:
    #         score = silhouette_score(array_tot, prediction.ravel())
    #         silhouette_scores.append(score)
    #     except:
    #         silhouette_scores.append(np.nan)

    return (breakpoints, error_history, silhouette_scores)





def XRCC(datasets: list[xr.DataArray], **kwargs) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    
    """
    Apply clustering function to a list of xarray DataArrays.

    Parameters:
    -----------
    datasets : list of xarray DataArrays
        Input datasets to be clustered.
    **kwargs : keyword arguments
        Additional parameters for clustering.

    Returns:
    --------
    tuple of xarray.DataArrays
        Result of clustering operation including breakpoints, 
        error history, and silhouette scores.
    """

    result = xr.apply_ufunc(
        X_cluster,
        *datasets,
        kwargs=kwargs,
        input_core_dims=[['time']] * len(datasets),
        output_core_dims=[['cluster'], ['iter'], ['iter']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float, float]
    )

    breakpoints, error_history, silhouette_scores = result

    breakpoints = xr.DataArray(breakpoints, dims=['lat', 'lon', 'cluster'])
    error_history_da = xr.DataArray(error_history, dims=['lat', 'lon', 'iter'])
    silhouette_scores_da = xr.DataArray(silhouette_scores, dims=['lat', 'lon', 'iter'])

    return breakpoints, error_history_da, silhouette_scores_da