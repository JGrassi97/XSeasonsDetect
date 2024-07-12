import numpy as np
import xarray as xr
from .radially_constrained_cluster import Radially_Constrained_Cluster  # Adjust import as per your package structure
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
    np.ndarray
        Array of breakpoints from clustering.
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
        grid_points_var = np.reshape(grid_points_var, (365, 32), order='F')
        
        if np.isnan(grid_points_var).any():
            return np.full(n_seas, np.nan)
        
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
    breakpoints = model.breakpoints

    return breakpoints


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
    xarray.DataArray
        Result of clustering operation.
    """
    result = xr.apply_ufunc(
        X_cluster,
        *datasets,
        kwargs=kwargs,
        input_core_dims=[['time']] * len(datasets),
        output_core_dims=[['cluster']],
        vectorize=True,
        output_sizes={'cluster': kwargs.get('n_seas', 2)},
        dask='parallelized',
        output_dtypes=[float]
    )

    return result


def compute_silhouette_scores(*grid_points, max_seasons=5, **kwargs):
    """
    Compute silhouette scores for different numbers of seasons (clusters).

    Parameters:
    -----------
    *grid_points : tuple of numpy arrays
        Input grid points variables to compute silhouette scores.
    max_seasons : int, optional
        Maximum number of seasons (clusters) to compute silhouette scores for.
    **kwargs : keyword arguments
        Additional parameters for clustering.

    Returns:
    --------
    xarray.DataArray
        DataArray with silhouette scores for each number of seasons.
        Dimensions: ('seasons',), Coordinates: ('seasons',).
    """
    silhouette_scores = []

    for n_seas in range(2, max_seasons + 1):
        arrays = []
        
        for grid_points_var in grid_points:
            grid_points_var = np.asarray(grid_points_var)
            grid_points_var = np.reshape(grid_points_var, (365, 32), order='F')
            
            if np.isnan(grid_points_var).any():
                silhouette_scores.append(np.nan)
                continue
            
            arrays.append(grid_points_var)
        
        combined_mask = ~np.any([np.all(np.isnan(arr), axis=0) for arr in arrays], axis=0)
        
        normalized_arrays = []
        for arr in arrays:
            array_tot = arr[:, combined_mask]
            array_tot = (array_tot - array_tot.min(axis=1).reshape(-1, 1)) / (array_tot.max(axis=1) - array_tot.min(axis=1)).reshape(-1, 1)
            normalized_arrays.append(array_tot)
        
        array_tot = np.concatenate(normalized_arrays, axis=1)

        # Initialize and fit the Radially_Constrained_Cluster model
        model_params = {
            'n_seas': n_seas,
            'n_iter': kwargs.get('iters', 20),
            'learning_rate': kwargs.get('learning_rate', 10),
            'min_len': kwargs.get('min_len', 30),
            'mode': kwargs.get('mode', 'single'),
            'starting_bp': kwargs.get('starting_bp', [165, 264])
        }

        model = Radially_Constrained_Cluster(data_to_cluster=array_tot, **model_params)
        model.fit()
        breakpoints = model.breakpoints

        # Compute silhouette score
        silhouette = silhouette_score(array_tot, model.get_index())
        silhouette_scores.append(silhouette)

    # Convert silhouette scores to xarray DataArray
    seasons = np.arange(2, max_seasons + 1)
    da_silhouette_scores = xr.DataArray(silhouette_scores, dims=('seasons',), coords={'seasons': seasons})

    return da_silhouette_scores



def find_optimal_seasons(silhouette_scores):
    """
    Find the number of seasons with the highest silhouette score.

    Parameters:
    -----------
    silhouette_scores : xarray.DataArray
        DataArray containing silhouette scores for different numbers of seasons.

    Returns:
    --------
    int
        Number of seasons with the highest silhouette score.
    """
    optimal_seasons = silhouette_scores['seasons'].values[np.argmax(silhouette_scores.values)]
    return optimal_seasons


def XRCC_silscore(datasets, max_seasons=5, **kwargs):
    """
    Apply clustering and compute silhouette scores for a list of xarray DataArrays.

    Parameters:
    -----------
    datasets : list of xarray DataArrays
        Input datasets to be clustered.
    max_seasons : int, optional
        Maximum number of seasons (clusters) to compute silhouette scores for.
    **kwargs : keyword arguments
        Additional parameters for clustering.

    Returns:
    --------
    tuple
        Tuple containing:
        - xarray.DataArray: Result of clustering operation.
        - xarray.DataArray: Silhouette scores for different numbers of seasons.
        - int: Optimal number of seasons with the highest silhouette score.
    """
    clustering_result = XRCC(datasets, **kwargs)

    # Extract numpy arrays from xarray DataArrays for silhouette score computation
    grid_points = tuple([da.values for da in datasets])

    silhouette_scores = compute_silhouette_scores(*grid_points, max_seasons=max_seasons)
    optimal_seasons = find_optimal_seasons(silhouette_scores)

    return clustering_result, silhouette_scores, optimal_seasons