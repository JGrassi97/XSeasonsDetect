import numpy as np
import xarray as xr
from .BRUTAL_radially_constrained_cluster import Radially_Constrained_Cluster  # Adjust import as per your package structure
from sklearn.metrics import silhouette_score


def X_cluster(*grid_points, **kwargs):
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
    iters = kwargs.get('iters', 20)  
    n_seas = kwargs.get('n_seas', 2)
    learning_rate = kwargs.get('learning_rate', 10)
    min_len = kwargs.get('min_len', 30)
    mode = kwargs.get('mode', 'single')
    starting_bp = kwargs.get('starting_bp', [165, 264])

    arrays = []
    
    for grid_points_var in grid_points:
        grid_points_var = np.asarray(grid_points_var)
        grid_points_var = np.reshape(grid_points_var, (365, int(grid_points_var.size/365)), order='F')
        
        if np.isnan(grid_points_var).any():
            return (np.full(n_seas, np.nan), np.full(iters, np.nan), np.full(iters, np.nan))
        
        arrays.append(grid_points_var)
    
    combined_mask = ~np.any([np.all(np.isnan(arr), axis=0) for arr in arrays], axis=0)
    
    normalized_arrays = []
    for arr in arrays:
        array_tot = arr[:, combined_mask]
        array_tot = (array_tot - array_tot.min(axis=1).reshape(-1, 1)) / (array_tot.max(axis=1) - array_tot.min(axis=1)).reshape(-1, 1)
        normalized_arrays.append(array_tot)
    
    array_tot = np.concatenate(normalized_arrays, axis=1)

    # Initialize and fit the Radially_Constrained_Cluster model
    model = Radially_Constrained_Cluster(data_to_cluster=array_tot,
                                         n_seas=n_seas,
                                         n_iter=iters,
                                         learning_rate=learning_rate,
                                         min_len=min_len,
                                         mode=mode,
                                         starting_bp=starting_bp)
    model.fit()

    # Returning final breakpoints
    breakpoints = model.breakpoints

    # Returning list of breakpoints
    breakpoint_history = model.breakpoint_history

    # Returning lists for diagnostic
    error_history = model.error_history
    prediction_history = model.prediction_history

    # Calculate silhouette scores for each set of breakpoints
    silhouette_scores = []
    for pred in prediction_history:
        try:
            # Calculate silhouette score
            score = silhouette_score(array_tot, pred)
            silhouette_scores.append(score)
        except:
            silhouette_scores.append(np.nan)

    return (breakpoints, error_history, silhouette_scores)

import xarray as xr

def XRCC(datasets, **kwargs):
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
        output_core_dims=[['cluster'] ,['iter'], ['iter']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float,float, float]
    )

    breakpoints, error_history, silhouette_scores = result

    error_history_da = xr.DataArray(error_history, dims=['lat', 'lon', 'iter'])
    silhouette_scores_da = xr.DataArray(silhouette_scores, dims=['lat', 'lon', 'iter'])

    return breakpoints, error_history_da, silhouette_scores_da


# import numpy as np
# import xarray as xr
# from .radially_constrained_cluster import Radially_Constrained_Cluster  # Adjust import as per your package structure
# from sklearn.metrics import silhouette_score

# def X_cluster(*grid_points, **kwargs):
    
#     """
#     Perform clustering on multiple variables from grid points.

#     Parameters:
#     -----------
#     *grid_points : tuple of numpy arrays
#         Input grid points variables to be clustered.
#     **kwargs : keyword arguments
#         Additional parameters for clustering.

#     Returns:
#     --------
#     np.ndarray
#         Array of breakpoints from clustering.
#     """
#     iters = kwargs.get('iters', 20)  
#     n_seas = kwargs.get('n_seas', 2)
#     learning_rate = kwargs.get('learning_rate', 10)
#     min_len = kwargs.get('min_len', 30)
#     mode = kwargs.get('mode', 'single')
#     starting_bp = kwargs.get('starting_bp', [165, 264])

#     arrays = []
    
#     for grid_points_var in grid_points:
#         grid_points_var = np.asarray(grid_points_var)
#         grid_points_var = np.reshape(grid_points_var, (365, int(grid_points_var.size/365)), order='F')
        
#         if np.isnan(grid_points_var).any():
#             return np.full(n_seas, np.nan)
        
#         arrays.append(grid_points_var)
    
#     combined_mask = ~np.any([np.all(np.isnan(arr), axis=0) for arr in arrays], axis=0)
    
#     normalized_arrays = []
#     for arr in arrays:
#         array_tot = arr[:, combined_mask]
#         array_tot = (array_tot - array_tot.min(axis=1).reshape(-1, 1)) / (array_tot.max(axis=1) - array_tot.min(axis=1)).reshape(-1, 1)
#         normalized_arrays.append(array_tot)
    
#     array_tot = np.concatenate(normalized_arrays, axis=1)

#     # Initialize and fit the Radially_Constrained_Cluster model
#     model = Radially_Constrained_Cluster(data_to_cluster=array_tot,
#                                          n_seas=n_seas,
#                                          n_iter=iters,
#                                          learning_rate=learning_rate,
#                                          min_len=min_len,
#                                          mode=mode,
#                                          starting_bp=starting_bp)
#     model.fit()

#     # Returning final breakpoints
#     breakpoints = model.breakpoints

#     # Returning lists for diagnostic
#     breakpoint_history = model.breakpoint_history
#     error_history = model.error_history
#     prediction_history = model.prediction_history


#     # Calculate silhouette scores for each set of breakpoints
#     silhouette_scores = []

#     for pred in prediction_history:

#         try:
#             # Calculate silhouette score
#             score = silhouette_score(array_tot, pred)
#             silhouette_scores.append(score)
        
#         except:
#             silhouette_scores.append(np.nan)
        

    
#     print(breakpoints, error_history, silhouette_scores)


#     return breakpoints, error_history, silhouette_scores



# def XRCC(datasets, **kwargs):

#     """
#     Apply clustering function to a list of xarray DataArrays.

#     Parameters:
#     -----------
#     datasets : list of xarray DataArrays
#         Input datasets to be clustered.
#     **kwargs : keyword arguments
#         Additional parameters for clustering.

#     Returns:
#     --------
#     tuple of xarray.DataArrays
#         Result of clustering operation including breakpoints, 
#         breakpoint history, error history, prediction history, 
#         and silhouette scores.
#     """
#     result = xr.apply_ufunc(
#         X_cluster,
#         *datasets,
#         kwargs=kwargs,
#         input_core_dims=[['time']] * len(datasets),
#         output_core_dims=[['cluster'], ['iter'], ['iter']],
#         vectorize=True,
#         # output_sizes={'cluster': kwargs.get('n_seas', 2), 'iter': kwargs.get('iters', 20)},
#         dask='parallelized',
#         output_dtypes=[float, float, float]
#     )

#     breakpoints, error_history, silhouette_scores = result


#     error_history_da = xr.DataArray(error_history, dims=['lat', 'lon', 'iter'])
#     silhouette_scores_da = xr.DataArray(silhouette_scores, dims=['lat', 'lon', 'iter'])

#     return breakpoints,  error_history_da, silhouette_scores_da


# # def XRCC(datasets, **kwargs):
# #     """
# #     Apply clustering function to a list of xarray DataArrays.

# #     Parameters:
# #     -----------
# #     datasets : list of xarray DataArrays
# #         Input datasets to be clustered.
# #     **kwargs : keyword arguments
# #         Additional parameters for clustering.

# #     Returns:
# #     --------
# #     xarray.DataArray
# #         Result of clustering operation.
# #     """
# #     result = xr.apply_ufunc(
# #         X_cluster,
# #         *datasets,
# #         kwargs=kwargs,
# #         input_core_dims=[['time']] * len(datasets),
# #         output_core_dims=[['cluster']],
# #         vectorize=True,
# #         output_sizes={'cluster': kwargs.get('n_seas', 2)},
# #         dask='parallelized',
# #         output_dtypes=[float]
# #     )

# #     return result




