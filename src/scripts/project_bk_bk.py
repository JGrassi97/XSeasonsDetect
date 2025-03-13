import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
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
from sklearn.linear_model import Perceptron


# Funzioni di visualizzazione e modelli
from visualization.custom_plots import day_of_year_to_date
from models.classifier import X_labels, train_perceptron, predict_custom

# Suppress warnings
warnings.filterwarnings("ignore")



def rolling_doy_complete(da, window_size=30):
    # Creiamo il giorno dell'anno (DOY) direttamente con Xarray per supportare calendari speciali
    doy = da.time.dt.dayofyear  
    zscore_da = xr.full_like(da, np.nan)  # Inizializziamo il dataset vuoto
    
    for d in range(1, 366):  
        # Selezioniamo tutti i valori corrispondenti a quel giorno dell'anno
        doy_mask = doy == d
        times_doy = da.time[doy_mask]

        
        # Se ci sono meno di 2 valori disponibili, saltiamo
        if len(times_doy) < 2:
            continue

        for i, t in enumerate(times_doy):
            # Definiamo la finestra mobile con gestione dei bordi
            start = max(0, i - window_size // 2)
            end = min(len(times_doy), i + window_size // 2 + 1)  

            # Selezioniamo la finestra
            window_doy = da.sel(time=times_doy[start:end])
            
            # Calcoliamo media e deviazione standard
            mean_doy = window_doy.mean(dim='time')

            # Normalizziamo evitando divisioni per zero
            zscore_da.loc[dict(time=t)] = (mean_doy)

    return zscore_da




def rolling_zscore_complete(da, window_size=30):
    times = da.time.values
    zscore_da = xr.full_like(da, np.nan)  # Creiamo un DataArray vuoto per i valori normalizzati
    
    for i, t in enumerate(times):
        # Definiamo i limiti della finestra
        start = max(0, i - window_size // 2)
        end = min(len(times), i + window_size // 2)

        # Selezioniamo i dati nella finestra
        window = da.isel(time=slice(start, end))
        
        # Calcoliamo la media e la deviazione standard
        min_wind = window.min(dim='time')
        max_wind = window.max(dim='time')

        # Evitiamo divisioni per zero
        try:
            #zscore_da.loc[dict(time=t)] = (da.sel(time=t) - mean) / std
            zscore_da.loc[dict(time=t)] = (da.sel(time=t) - min_wind) / (max_wind - min_wind)
        except:
            zscore_da.loc[dict(time=t)] = 0  # Se la deviazione standard è 0, assegniamo 0

    return zscore_da




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
        dat = xr.open_dataset(path)[code]
        dat = rolling_doy_complete(dat, window_size=15)
        #dat = rolling_zscore_complete(dat, window_size=365)
        dataset_train.append(dat)

    if len(dataset_train) > 1:
        
        for j in range(1, len(dataset_train)):
            dataset_train[j]['time'] = dataset_train[0]['time']

    dataset_train = xr.merge(dataset_train).sel(time=slice('1971','2000'))

    dataset_train = (dataset_train - dataset_train.mean(dim='time')) / dataset_train.std(dim='time')
    #dataset_train = (dataset_train - dataset_train.min(dim='time')) / (dataset_train.max(dim='time') - dataset_train.min(dim='time'))
    array_train  = dataset_train.to_array().values.transpose().reshape(36,36,365,30*4, order='F')

    index_values = dates_clust.values
    index_values_expanded = np.expand_dims(index_values, axis=-1)  # (5,5,365,1) 
    array_train = np.concatenate((array_train, index_values_expanded), axis=-1)



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


        dataset_proj = []
        for path, code in zip(proj_paths, proj_codes):
            
            try:
                dat = xr.open_dataset(path)[code].mean('plev')
            except:
                dat = xr.open_dataset(path)[code]

            dat = rolling_doy_complete(dat, window_size=15)
            #dat = rolling_zscore_complete(dat, window_size=365)
            dataset_proj.append(dat)
        
    dataset_proj = xr.merge(dataset_proj)


    dataset_proj_0 = dataset_proj.sel(time=slice('1971','2000'))
    dataset_proj_1 = dataset_proj.sel(time=slice('2016','2045'))
    dataset_proj_2 = dataset_proj.sel(time=slice('2041','2070'))
    dataset_proj_3 = dataset_proj.sel(time=slice('2070','2099'))

    dataset_proj_0 = (dataset_proj_0 - dataset_proj_0.mean(dim='time')) / dataset_proj_0.std(dim='time')
    dataset_proj_1 = (dataset_proj_1 - dataset_proj_1.mean(dim='time')) / dataset_proj_1.std(dim='time')
    dataset_proj_2 = (dataset_proj_2 - dataset_proj_2.mean(dim='time')) / dataset_proj_2.std(dim='time')
    dataset_proj_3 = (dataset_proj_3 - dataset_proj_3.mean(dim='time')) / dataset_proj_3.std(dim='time')

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
        "MLPClassifier": MLPClassifier,
        "Perceptron" : Perceptron
    }

    if model_name not in MODELS:
        raise ValueError(f"Modello '{model_name}' non supportato!")
    
    # model = MODELS[model_name](**model_params)


    mse, r2, models, histories = train_perceptron(data, 120, 4, epochs=50, n_year_training=1)

    # dataset_model = xr.Dataset(
    #     {
    #         'mse': (('lat', 'lon'), mse),
    #         'r2': (('lat', 'lon'), r2),
    #         #'model': (('lat', 'lon'), models)
    #     },
    #     coords={
    #         'lat': dataset_train['lat'],
    #         'lon': dataset_train['lon']
    #     }
    # )


    


    # dataset_model.attrs['description'] = 'Model trained on the dataset with the labels obtained from XSeas_detect'
    # #dataset_model.attrs['model'] = model

    # dataset_model.mse.attrs['long_name'] = 'Mean Squared Error'
    # dataset_model.r2.attrs['long_name'] = 'R2 Score'
    # #dataset_model.model.attrs['long_name'] = 'Model'

    # dataset_model.to_netcdf(os.path.join(os.getcwd(),'results', 'files', f'{name}_{model_name}_projection_training_{scenario_model[0]}.nc'))




    predictions = []
    for dat in [dataset_proj_0, dataset_proj_1, dataset_proj_2, dataset_proj_3]:

        array_res  = dat.to_array().values.transpose().reshape(36,36,365,30*4, order='F')
        predictions.append(predict_custom(array_res, 120, models))



    predictions_tot = np.array(predictions).transpose((1,2,3,0)).reshape(36,36,365*4)
    print(predictions_tot.shape)

    # Lista degli anni desiderati
    years = [1985, 2030, 2055, 2085]

    # Genera le date per ciascun anno
    date_list = [pd.date_range(f"{year}-01-01", f"{year}-12-31") for year in years]

    # Concatena tutte le date in una sola lista
    all_dates = np.concatenate(date_list)


    predictions_xr = xr.Dataset(
        {
            'predictions': (('lat', 'lon', 'time'), predictions_tot)
        },
        coords={
            'time': all_dates,
            'lat': dataset_proj['lat'],
            'lon': dataset_proj['lon']
        }
    )

    predictions_xr.to_netcdf(os.path.join(os.getcwd(),'results', 'files', f'{name}_{model_name}_projection_prediction_{scenario_model[0]}.nc'))