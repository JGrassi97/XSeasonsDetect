import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import warnings
from tqdm import tqdm
import argparse
import os
import yaml

# Modelli e metriche
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, 
                              GradientBoostingClassifier, GradientBoostingRegressor, 
                              AdaBoostClassifier, BaggingClassifier, BaggingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# Funzioni di visualizzazione e modelli
from visualization.custom_plots import day_of_year_to_date
from models.classifier import X_labels, train, predict_custom

# Suppress warnings
warnings.filterwarnings("ignore")





def main():

    # Initialize the parser
    parser = argparse.ArgumentParser(description='Perform a projection analysis with XSeasonsDetect')
    
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
    

    # --GET TRAINING INFO FROM XSeas_cluster .yaml file 

    # Load settings 
    train_paths = settings['parameters']['variables']
    train_codes = settings['parameters']['variable_code']
    n_seasons = settings['parameters']['n_seasons']
    name = settings['name']

    # Load the files with dates for the training
    dataset = xr.open_dataset(os.path.join(os.getcwd(),'results', 'files', f'{name}.nc'))

    # Create the labels for the training
    lables_param = {
        'n_seas': n_seasons
    }
    dates_clust = X_labels(dataset['breakpoints'], **lables_param)

    # Load the training dataset
    dataset_train = []

    for path, code in zip(train_paths, train_codes):
        dataset_train.append(xr.open_dataset(path)[code])

    if len(dataset_train) > 1:
        
        for j in range(1, len(dataset_train)):
            dataset_train[j]['time'] = dataset_train[0]['time']

    dataset_train = xr.merge(dataset_train)


    #dataset_train = (dataset_train - dataset_train.min(dim='time')) / (dataset_train.max(dim='time') - dataset_train.min(dim='time'))
    dataset_train = (dataset_train - dataset_train.mean(dim='time')) / dataset_train.std(dim='time')

    index_values = dates_clust.values
    index_values = np.tile(index_values, 57).transpose((2, 0, 1))
    dataset_train['labels'] = (('time', 'lat', 'lon'), index_values)


    array_train  = dataset_train.to_array().values.transpose()

    # Project dataset

    # Load settings 
    variables = settings['parameters_projetions']['variables']
    proj_codes = settings['parameters_projetions']['variable_code']
    proj_path = settings['parameters_projetions']['model_path'][0]

    n_features = len(variables)

    # Creazione dei percorsi per i file di proiezione
    proj_paths = []
    for scenario in ['historical','ssp585']:
        for variable in variables:
            path = os.path.join(proj_path, scenario, variable, 'final.nc')
            proj_paths.append(path)

    dataset_proj = xr.merge([xr.open_dataset(path)[code] for path, code in zip(proj_paths, proj_codes)]).mean('plev')


    #dataset_res = (dataset_res - dataset_res.min(dim='time')) / (dataset_res.max(dim='time') - dataset_res.min(dim='time'))
    dataset_proj_norm = dataset_proj.sel(time=slice('1961', '2019'))
    dataset_proj = (dataset_proj - dataset_proj_norm.mean(dim='time')) / dataset_proj_norm.std(dim='time')

    data = array_train

    # MODELS 

    # Neural Network Classifier (MLP)

    model_name = settings['parameters_projetions']["model"]["name"]
    model_params = settings['parameters_projetions']["model"].get("params", {})

    scenario_model = settings['parameters_projetions']['scenario_model']

    # Mappatura dei modelli disponibili
    MODELS = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "SVC": SVC,
        "KNeighborsClassifier": KNeighborsClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "GaussianNB": GaussianNB,
        "AdaBoostClassifier": AdaBoostClassifier,
        "MLPClassifier": MLPClassifier
    }

    if model_name not in MODELS:
        raise ValueError(f"Modello '{model_name}' non supportato!")
    
    model = MODELS[model_name](**model_params)



    mse, r2, models = train(data, n_features, model=model)

    dataset_model = xr.Dataset(
        {
            'mse': (('lat', 'lon'), mse),
            'r2': (('lat', 'lon'), r2),
            #'model': (('lat', 'lon'), models)
        },
        coords={
            'lat': dataset_train['lat'],
            'lon': dataset_train['lon']
        }
    )


    


    dataset_model.attrs['description'] = 'Model trained on the dataset with the labels obtained from XSeas_detect'
    #dataset_model.attrs['model'] = model

    dataset_model.mse.attrs['long_name'] = 'Mean Squared Error'
    dataset_model.r2.attrs['long_name'] = 'R2 Score'
    #dataset_model.model.attrs['long_name'] = 'Model'

    dataset_model.to_netcdf(os.path.join(os.getcwd(),'results', 'files', f'{name}_{model_name}_projection_training_{scenario_model[0]}.nc'))




    array_res  = dataset_proj.to_array().values.transpose()
    predictions = predict_custom(array_res, n_features, models)



    # Put predictions in a xarray.Dataset
    predictions_xr = xr.DataArray(
        predictions,
        dims=["lat", "lon", "time"],
        coords={
            
            "lat": dataset_proj.lat,
            "lon": dataset_proj.lon,
            "time": dataset_proj.time,
        },
        name="predictions",
    )

    predictions_xr.to_netcdf(os.path.join(os.getcwd(),'results', 'files', f'{name}_{model_name}_projection_prediction_{scenario_model[0]}.nc'))