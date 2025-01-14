import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics import silhouette_score
import numpy as np
from shapely.geometry import mapping
import pandas as pd
import os
import matplotlib.animation as animation
import yaml
import argparse

from models.X_RCC import XRCC#, XRCC_silhouette
#from visualization.custom_plots import standard_format, day_of_year_to_date, standard_format_single, plot_seasons_bk_results
from visualization.visual_cluster_results import plot_seasons_bk_results

import warnings
warnings.filterwarnings('ignore')



def main():

    # Initialize the parser
    parser = argparse.ArgumentParser(description='Perform a clustering analysis with XSeasonsDetect')
    
    # Add the arguments - Name of the project
    parser.add_argument('--exp_setting', type=str, help='Name of the .yaml file with the setting for the experiment', required=True)
    parser.add_argument('--start', type=int, help='Year of start', required=False)
    parser.add_argument('--end', type=int, help='Year of end', required=False)

    # Parse the arguments
    args = parser.parse_args()

    setting_file = os.path.join(os.getcwd(),'config', args.exp_setting)
    if not os.path.exists(setting_file):
        raise FileNotFoundError(f'The file {setting_file} does not exist')

    # Load the settings
    with open(setting_file, 'r') as file:
        print('Loading settings')
        settings = yaml.load(file, Loader=yaml.FullLoader)
    
    # Load variables as stored in the settings['parameters']
    variables = settings['parameters']['variables']
    variable_codes = settings['parameters']['variable_code']

    print(variables, variable_codes)
    datasets = []


    for variable, code in zip(variables, variable_codes):

        dataset = xr.open_mfdataset(variable)[code].load()
        print(variable)
        if args.start is not None and args.end is not None:
            dataset = dataset.sel(time=slice(f'{args.start}', f'{args.end}'))
        datasets.append(dataset)
    
    # Standardize the datasets time as the one of the first dataset
    for i in range(1, len(datasets)):
        datasets[i]['time'] = datasets[0]['time']

    # Load the clustering parameters
    n_iters = settings['parameters']['n_iters']
    n_seasons = settings['parameters']['n_seasons']
    learning_rate = settings['parameters']['learning_rate']
    min_len = settings['parameters']['min_len']
    starting_bp = settings['parameters']['starting_breakpoints']
    weights = settings['parameters']['weights']

    clustering_params = {
        'iters': n_iters,
        'n_seas': n_seasons,
        'learning_rate': learning_rate,
        'min_len': min_len,
        'mode': 'single',
        'starting_bp': starting_bp,
        'weights': weights
    }

    breakpoints, error_history_da, silhouette_scores_da = XRCC(datasets, **clustering_params)

    # Merge the results into a single xr.DataSet [error_history and silhouette_scores have an additional dimension 'iters']
    results = xr.merge([breakpoints.rename('breakpoints'), error_history_da.rename('error_history'), silhouette_scores_da.rename('silhouette_scores')])

    name = settings['name']

    # Save the breakpoints
    if args.start is not None and args.end is not None:
        results.to_netcdf(os.path.join(os.getcwd(),'results', 'files', f'{name}_{args.start}_{args.end}.nc'))
    else:
        results.to_netcdf(os.path.join(os.getcwd(),'results', 'files', f'{name}.nc'))