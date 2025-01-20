import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm



def get_prediction(b, **kwargs):

    n_seas = kwargs['n_seas']

    # Converting breakpoints in a time series 
    prediction = np.zeros(365)

    try:
        idx = generate_season_idx(b, n_seas)
        for i in range(n_seas):
            prediction[idx[i].astype(int)] = i

        # Return the prediction as a 1D array
        return prediction.astype(int)

    except:
        return np.nan * np.ones(365)

def generate_season_idx(b, n_seas):

    idx = []

    if n_seas == 1:
        idx.append(np.arange(0, 365, 1))

    else:
        for i in np.arange(-1, n_seas-1,1):
            if b[i]>b[i+1]:
                idx_0 = np.arange(b[i], 365, 1)
                idx_1 = np.arange(0, b[i+1], 1)
                idx.append(np.concatenate((idx_0, idx_1), axis=None))

            else:
                idx.append(np.arange(b[i], b[i+1],1))

    return idx



# Applicazione su dati xarray
#breakpoints = dataset['breakpoints']  # Dataset con dimensioni (lat, lon, cluster)
def X_labels(breakpoints: xr.DataArray, **kwargs):

    dates_clust = xr.apply_ufunc(
        get_prediction, 
        breakpoints,  # Input
        kwargs=kwargs,  # Numero di stagioni
        vectorize=True,  # Itera su ciascun elemento
        dask="parallelized",  # Usa Dask se disponibile
        input_core_dims=[["cluster"]],  # Le dimensioni 'cluster' sono nell'input
        output_core_dims=[["dayofyear"]],  # L'output avrà 'dayofyear'
        dask_gufunc_kwargs={"output_sizes": {"dayofyear": 365}},  # Dimensione 'dayofyear' con 365 giorni
        output_dtypes=[int],  # Tipo di dati in output
        keep_attrs=True
    )


    return dates_clust







def train_single(X, Y, model=None):
    """
    Generalized training function that supports different models.

    Parameters:
    - X: np.array or pd.DataFrame, feature matrix
    - Y: np.array or pd.Series, target variable
    - model: scikit-learn model instance (default: LogisticRegression)

    Returns:
    - test_mse: float, Mean Squared Error on test set
    - test_r2: float, R^2 Score on test set
    - trained_model: trained model instance
    """
    
    # Check for NaN values
    if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        return np.nan, np.nan, None

    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Default model: Logistic Regression (if none is provided)
    if model is None:
        model = LogisticRegression(random_state=0)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    return test_mse, test_r2, model




def train(data, model = None):

    r2 = np.zeros((np.shape(data)[0], np.shape(data)[1]))
    mse = np.zeros((np.shape(data)[0], np.shape(data)[1]))
    models = np.zeros((np.shape(data)[0], np.shape(data)[1]), dtype=object)

    # Apply the function to all the point in the first 2 dimension
    for j in tqdm(range(np.shape(data)[0])):
        for i in range(np.shape(data)[1]):
            
            x = data[j,i,:,0:2]
            y = data[j,i,:,3]

            res = train_single(x, y, model=model)

            mse[i,j] = res[0]
            r2[i,j] = res[1]
            models[i,j] = res[2]
        
    return mse, r2, models




def predict(X, model):    
    if np.any(np.isnan(X)):
        return np.nan

    return model.predict(X)



def predict_custom(array_res, datarray_model):

    predictions = np.zeros((np.shape(array_res)[0], np.shape(array_res)[1], np.shape(array_res)[2]))

    models = datarray_model

    for i in tqdm(range(np.shape(array_res)[0])):
        for j in range(np.shape(array_res)[1]):
            
            model = models[i,j]
            x = array_res[j,i,:,0:2]

            try:
                predictions[i,j,:] = predict(x, model)
            except:
                predictions[i,j,:] = np.nan
    

    return predictions

