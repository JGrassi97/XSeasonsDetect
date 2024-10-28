import xarray as xr
import os
import numpy as np
import cftime 
import rioxarray
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import mapping
import dask 
import pandas as pd



def sel_years(dataset, start, end):

    dataset = dataset.sel(time=slice(start, end)).chunk(dict(time=-1))
    return dataset


def interpolate_na(dataset):

    try:
        ### FILLING NA IN DATASET TP
        # Verifica l'asse temporale del dataset
        time = dataset.time

        # Crea un indice completo con frequenza oraria
        complete_time_index = pd.date_range(start=time.min().item(), end=time.max().item(), freq='D')

    except:
        dataset = dataset.convert_calendar('standard')
        ### FILLING NA IN DATASET TP
        # Verifica l'asse temporale del dataset
        time = dataset.time

        # Crea un indice completo con frequenza oraria
        complete_time_index = pd.date_range(start=time.min().item(), end=time.max().item(), freq='D')
        
    # Reindicizza il dataset per includere tutte le date, anche quelle mancanti
    ds_reindexed = dataset.reindex(time=complete_time_index)

    # Interpola i dati per riempire i valori mancanti
    dataset = dataset = ds_reindexed.interpolate_na(dim='time', method='linear')

    return dataset



def clean_cut(dataset, boundary = None, window = None, remove_empty = True):

    # Converting calendar and removing useless dimensions
    dataset = dataset.convert_calendar('noleap')

    try:
        dataset = dataset.drop_dims('bnds')
    except:
        pass

    if boundary is not None:
        # Setting the datasets for masking
        dataset.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        dataset.rio.write_crs("epsg:4326", inplace=True)

        # Masking the datasets
        dataset = dataset.rio.clip(boundary.geometry.apply(mapping), boundary.crs, drop=True)

    if window is not None:
        dataset = dataset.rolling(time=window, center=True).mean()

    
    if remove_empty:
        start_year = dataset.time.dt.year.min().values
        end_year = dataset.time.dt.year.max().values
        dataset = dataset.sel(time=slice(str(start_year+1),str(end_year-1)))

    return dataset


def remap_cdo(in_file, out_file, grid_file):

    os.system(f'cdo remapbil,{grid_file} {in_file} {out_file}')


def standard_preprocess(in_path, temp_path, out_path, start_year, end_year, grid_file, boundary = None, window = None, remove_empty = True, out_filename = 'final'):

    remap_cdo(f'{in_path}/*.nc', f'{temp_path}/temp.nc', grid_file)

    raw_dataset = xr.open_mfdataset(f'{temp_path}/temp.nc')

    dataset_work = sel_years(raw_dataset, str(start_year),str(end_year))
    dataset_work = interpolate_na(dataset_work)
    dataset_work = clean_cut(dataset_work, boundary, window, remove_empty)

    dataset_work.to_netcdf(f'{out_path}/{out_filename}.nc')

    os.remove(f'{temp_path}/temp.nc')