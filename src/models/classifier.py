import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical



def get_prediction(b, **kwargs):
    n_seas = kwargs['n_seas']
    prediction = np.zeros(365)

    try:
        idx = generate_season_idx(b, n_seas)
        for i in range(n_seas):
            prediction[idx[i].astype(int)] = i

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



def X_labels(breakpoints: xr.DataArray, **kwargs):
    dates_clust = xr.apply_ufunc(
        get_prediction, 
        breakpoints, 
        kwargs=kwargs,  
        vectorize=True,  
        dask="parallelized",
        input_core_dims=[["cluster"]], 
        output_core_dims=[["dayofyear"]], 
        dask_gufunc_kwargs={"output_sizes": {"dayofyear": 365}},
        output_dtypes=[int], 
        keep_attrs=True
    )
    return dates_clust



def build_model(input_shape, n_seas):
    model = Sequential()
    model.add(Dense(n_seas, input_dim=input_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_perceptron(data, n_features, n_seas, n_year_training=50, epochs=50):

    r2 = np.zeros((np.shape(data)[0], np.shape(data)[1]))
    mse = np.zeros((np.shape(data)[0], np.shape(data)[1]))
    models = np.zeros((np.shape(data)[0], np.shape(data)[1]), dtype=object)
    histories = np.zeros((np.shape(data)[0], np.shape(data)[1], epochs), dtype=object)

    for j in tqdm(range(np.shape(data)[0])):
        for i in range(np.shape(data)[1]):

            try:

                x = data[j,i,:,0:n_features]
                y = data[j,i,:,n_features]

                y = to_categorical(y)

                x_train, x_test = x[:365*n_year_training], x[365*n_year_training:]
                y_train, y_test = y[:365*n_year_training], y[365*n_year_training:]

                mod = build_model(n_features, y.shape[1])

                history = mod.fit(x, y, epochs=epochs, batch_size=52, verbose=False)
                histories[j,i,:] = history.history['accuracy']

                # mse[i,j] = np.nan
                # r2[i,j] = np.nan
                models[j,i] = mod
            
            except:
                
                mse[j,i] = np.nan
                r2[j,i] = np.nan
                models[j,i] = None
        
    return mse, r2, models, histories



def predict_custom(array_res, n_features, datarray_model):

    predictions = np.zeros((np.shape(array_res)[0], np.shape(array_res)[1], np.shape(array_res)[2]))

    models = datarray_model

    for j in tqdm(range(np.shape(array_res)[0])):
        for i in range(np.shape(array_res)[1]):
            
            model = models[j,i]
            x = array_res[j,i,:,0:n_features]

            try:
                predictions[j,i,:] = model.predict(x, verbose=False).argmax(axis=1)
            except:
                predictions[j,i,:] = np.nan

    return predictions
