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

    # Parse the arguments
    args = parser.parse_args()

    # Load the settings
    with open(os.path.join(os.getcwd(),'config', args.exp_setting), 'r') as file:
        settings = yaml.load(file, Loader=yaml.FullLoader)
    
    # Load variables as stored in the settings['parameters']
    variables = settings['parameters']['variables']
    variable_codes = settings['parameters']['variable_code']

    datasets = []

    for variable, code in zip(variables, variable_codes):
        dataset = xr.open_mfdataset(f'../data/preprocessed/ERA5/{variable}/final.nc')[code].load()
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

    clustering_params = {
        'iters': n_iters,
        'n_seas': n_seasons,
        'learning_rate': learning_rate,
        'min_len': min_len,
        'mode': 'single',
        'starting_bp': starting_bp,
    }

    breakpoints, error_history_da, silhouette_scores_da = XRCC(datasets, **clustering_params)

    name = settings['name']

    # Save the breakpoints
    breakpoints.to_netcdf(os.path.join(os.getcwd(),'results', 'files', f'{name}.nc'))